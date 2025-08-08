import { Injectable } from '@nestjs/common';
import { checkpointer, getCollection, MESSAGES_THREAD, USER_THREADS_COLLECTION } from 'src/db/mongodbSaver';
import { TavilySearch } from "@langchain/tavily";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { SystemMessage } from '@langchain/core/messages';
import { PromptTemplate } from "@langchain/core/prompts";
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { StateGraph, Annotation, START, addMessages } = require("@langchain/langgraph");
const dotenv = require("dotenv");
dotenv.config();

const SYSTEM_PROMPT = `
You are a helpful assistant.
You can answer questions, provide information, and assist with various tasks. 
If you don't know the answer, you can search the web for information.
Try to craft your response in markdown format, using appropriate headings, lists, and links to enhance readability.
Be detailed and thorough in your responses, ensuring that the user receives comprehensive information.
Summary from previous conversation: {summary}
`;

const SUMMARY_PROMPT = `
You are a helpful assistant.
You will be provided with a list of messages from a chat thread.
Your task is to summarize the conversation in a concise manner, highlighting the key points and main topics discussed.
Ensure that the summary captures the essence of the conversation without losing important details.

messages: {messages}
`;

const IF_SUMMARY_EXIST_PROMPT = `
You are a helpful assistant.
You will be provided with a list of messages from a chat thread.
Use the existing summary and messages to generate a concise summary of the conversation.
Ensure that the summary captures the essence of the conversation without losing important details.

messages: {messages}
summary: {summary}
`;

const MAX_MESSAGES = process.env.MAX_MESSAGES ? parseInt(process.env.MAX_MESSAGES) : 10;

type Chat = {
    id: string,
    type: "user" | "assistant",
    text: string,
    isWebSearch: boolean,
    urls: string[],
}

const model = new ChatGoogleGenerativeAI({
    model: 'gemini-2.0-flash-001',
});

@Injectable()
export class ChatService {
    graph;

    async compileChatGraph() {
        const TavilySearchTool = new TavilySearch({
            maxResults: 3
        });

        const tools = [TavilySearchTool];
        const modelWithTools = model.bindTools(tools);
        const toolNode = new ToolNode(tools);

        const GraphState = Annotation.Root({
            messages: Annotation({
                reducer: addMessages,
            }),
            summary: Annotation(),
        });

        // Utility to inject system prompt with summary
        const injectSystemPrompt = async (messages, summary) => {
            const systemPromptTemplate = PromptTemplate.fromTemplate(SYSTEM_PROMPT);
            const formatted = await systemPromptTemplate.format({ summary });
            const systemMessage = new SystemMessage(formatted);
            return [systemMessage, ...messages.filter(m => !(m.lc_id && m.lc_id.includes("SystemMessage")))];
        };

        // Agent logic
        const agent = async ({ messages, summary }) => {
            const humanMessages = messages.filter(m => m.lc_id.includes("HumanMessage"));

            // Summarize if too many messages
            if (humanMessages.length > MAX_MESSAGES) {
                const nonSystemMessages = messages.filter(m => !(m.lc_id && m.lc_id.includes("SystemMessage")));
                const msgsToSummarize = nonSystemMessages.slice(0, nonSystemMessages.length - MAX_MESSAGES);
                const summarizerPrompt = PromptTemplate.fromTemplate(summary ? IF_SUMMARY_EXIST_PROMPT : SUMMARY_PROMPT);
                const promptInput = summary
                    ? { messages: msgsToSummarize, summary }
                    : { messages: msgsToSummarize };

                const prompt = await summarizerPrompt.format(promptInput as any);
                const summaryResponse = await model.invoke(prompt, {
                    tags: ['internal_summary'],
                });
                summary = summaryResponse.content;

                // Keep only last few messages
                messages = messages.slice(-MAX_MESSAGES);
            }

            // Inject updated system prompt
            messages = await injectSystemPrompt(messages, summary);

            const response = await modelWithTools.invoke(messages);

            return {
                messages: messages.concat(response),
                summary,
            };
        };

        const workflow = new StateGraph(GraphState)
            .addNode("agent", agent)
            .addNode("tools", toolNode)
            .addEdge(START, "agent")
            .addConditionalEdges("agent", toolsCondition)
            .addEdge("tools", "agent");

        this.graph = workflow.compile({
            checkpointer,
        });

        console.log("✅ Chat graph compiled successfully");
    }

    async getChatGraph() {
        if (!this.graph) {
            await this.compileChatGraph();
        }
        return this.graph;
    }

    async getMessages(threadId: string) {
        const collection = getCollection(MESSAGES_THREAD);
        const threadDoc = await collection.findOne({ threadId });
        return threadDoc?.chats || []
    }

    async createOrUpdateUserThread({ userId, email, name, threadId, isNewThread, message }: { userId: string; email: string, name: string, threadId: string, isNewThread: boolean, message: string }) {
        const collection = getCollection(USER_THREADS_COLLECTION);
        const title = isNewThread ? await this.getConversationTitle(message) : undefined;
        await collection.updateOne(
            { threadId },
            {
                $set: {
                    userId,
                    email,
                    name,
                    threadId,
                    updatedAt: new Date(),
                },
                $setOnInsert: {
                    createdAt: new Date(),
                    ...(title ? { title } : {})
                }
            },
            { upsert: true }
        );
        return title;
    }

    async pushMessages({ threadId, userId, chat }: { threadId: string, userId: string, chat: Chat }) {
        const collection = getCollection(MESSAGES_THREAD);
        await collection.updateOne(
            { threadId },
            [
                {
                    $set: {
                        userId,
                        threadId,
                        updatedAt: new Date(),
                        feedback: { $ifNull: ["$feedback", ""] },
                        createdAt: { $ifNull: ["$createdAt", new Date()] },
                        chats: { $concatArrays: [{ $ifNull: ["$chats", []] }, [chat]] }
                    }
                }
            ],
            { upsert: true }
        );
    }

    async getConversationTitle(message: string) {
        const prompt = PromptTemplate.fromTemplate("Generate a concise title for the following conversation: {message}. Only generate one title and only return that tile and nothing else");
        const formattedPrompt = await prompt.format({ message });

        const response = await model.invoke(formattedPrompt);
        return response.content;
    }

    async getChatHistory(userId: string) {
        const collection = getCollection(USER_THREADS_COLLECTION);
        const threads = await collection.find({ userId }, { sort: { "updatedAt": -1 } }).toArray();
        return threads.map(thread => ({
            threadId: thread.threadId,
            title: thread.title || "Untitled Conversation",
            updatedAt: thread.updatedAt,
            createdAt: thread.createdAt,
        }));
    }

    async updateFeedback(threadId, messageId, feedback) {
        const collection = getCollection(MESSAGES_THREAD);
        await collection.updateOne({
            threadId,
            "chats.id": messageId
        }, {
            $set: {
                "chats.$.feedback": feedback
            }
        })
    }
}
