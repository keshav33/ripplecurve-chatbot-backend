import { Injectable } from '@nestjs/common';
import { checkpointer, getCollection, MESSAGES_THREAD, USER_THREADS_COLLECTION } from 'src/db/mongodbSaver';
import { TavilySearch } from "@langchain/tavily";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { SystemMessage } from '@langchain/core/messages';
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { StateGraph, Annotation, START, addMessages, END } from "@langchain/langgraph";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
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

const FILE_QA_PROMPT = `
You are a helpful assistant.
You will be provided context and a question. Based on the context and question, answer the question.
Do not make up information by yourself and always refer the context.
conext: {context}
question: {question}
`

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
            summary: Annotation<string>,
            fileId: Annotation<string>,
            fileType: Annotation<string>,
        });

        // Utility to inject system prompt with summary
        const injectSystemPrompt = async (messages, summary) => {
            const systemPromptTemplate = PromptTemplate.fromTemplate(SYSTEM_PROMPT);
            const formatted = await systemPromptTemplate.format({ summary });
            const systemMessage = new SystemMessage(formatted);
            return [
                systemMessage,
                ...messages.filter(m => !(m.lc_id && m.lc_id.includes("SystemMessage")))
            ];
        };

        // ============= Primary Agent Logic =============
        const primaryAgentWithTools = async ({ messages, summary }) => {
            // Injected messages & summary have already been handled in wrapper
            const response = await modelWithTools.invoke(messages);
            return {
                messages: messages.concat(response),
                summary,
            };
        };

        // ============= PDF Agent Logic =============
        const pdfAgent = async ({ messages, fileId }) => {
            const loader = new PDFLoader(__dirname + `/uploads/${fileId}.pdf`);
            const docs = await loader.load();
            const splitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 100,
            });
            const splitDocs = await splitter.splitDocuments(docs);
            const vectorStore = await FaissStore.fromDocuments(
                splitDocs,
                new GoogleGenerativeAIEmbeddings()
            );
            const question = messages[messages.length - 1].content;
            const results = await vectorStore.similaritySearch(question, 3);
            const context = results.map(doc => doc.pageContent);
            const QAPrompt = PromptTemplate.fromTemplate(FILE_QA_PROMPT);
            const formatedPrompt = await QAPrompt.format({ question, context: context.join('\n') });
            const response = await model.invoke(formatedPrompt);
            return {
                messages: [response.content]
            };
        };

        // ============= Wrapper for Agents =============
        const agentWrapper = (agentFn) => {
            return async ({ messages, summary, fileId }) => {
                const humanMessages = messages.filter(m => m.lc_id.includes("HumanMessage"));

                // Summarize if too many messages
                if (humanMessages.length > MAX_MESSAGES) {
                    const nonSystemMessages = messages.filter(
                        m => !(m.lc_id && m.lc_id.includes("SystemMessage"))
                    );
                    const msgsToSummarize = nonSystemMessages.slice(
                        0,
                        nonSystemMessages.length - MAX_MESSAGES
                    );

                    const summarizerPrompt = PromptTemplate.fromTemplate(
                        summary ? IF_SUMMARY_EXIST_PROMPT : SUMMARY_PROMPT
                    );

                    const promptInput = summary
                        ? { messages: msgsToSummarize, summary }
                        : { messages: msgsToSummarize };

                    const prompt = await summarizerPrompt.format(promptInput as any);
                    const summaryResponse = await model.invoke(prompt, {
                        tags: ["internal_summary"],
                    });
                    summary = summaryResponse.content;

                    // Keep only last few messages
                    messages = messages.slice(-MAX_MESSAGES);
                }

                // Inject updated system prompt
                messages = await injectSystemPrompt(messages, summary);

                return agentFn({ messages, summary, fileId });
            };
        };

        // ============= Start Router =============
        const startRouter = (state) => {
            if (state.fileId) return "pdf_agent";
            return "agent";
        };

        // ============= Workflow Graph =============
        const workflow = new StateGraph(GraphState)
            // wrap primary agent and pdf agent so both get summarization + system prompt injection
            .addNode("agent", agentWrapper(primaryAgentWithTools))
            .addNode("tools", toolNode)
            .addNode("pdf_agent", agentWrapper(pdfAgent))
            // route at START so agent won't run for pdf requests
            .addConditionalEdges(START, startRouter)
            // IMPORTANT: agent -> toolCondition (agent's outgoing edge is conditional now)
            .addConditionalEdges("agent", (state) => {
                // call your toolsCondition function to decide the next node
                // it should return something like "tools" or END (or another node name)
                return toolsCondition(state as any);
            })
            // tools -> agent (tool runs and returns to agent)
            .addEdge("tools", "agent")
            // pdf_agent -> END
            .addEdge("pdf_agent", END);

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

    async deleteThread(threadId) {
        const messageThreadCollection = getCollection(MESSAGES_THREAD);
        const userThreadCollection = getCollection(USER_THREADS_COLLECTION);
        await Promise.all([
            userThreadCollection.deleteOne({ threadId }),
            messageThreadCollection.deleteOne({ threadId }),
            checkpointer.deleteThread(threadId)
        ])
        return;
    }
}
