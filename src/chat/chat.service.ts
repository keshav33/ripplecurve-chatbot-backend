import { Injectable } from '@nestjs/common';
import { checkpointer } from 'src/db/mongodbSaver';
import { TavilySearch } from "@langchain/tavily";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { SystemMessage } from '@langchain/core/messages';
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { StateGraph, Annotation, START, END, addMessages } = require("@langchain/langgraph");

const SystemPrompt = `
You are a helpful assistant.
You can answer questions, provide information, and assist with various tasks. 
If you don't know the answer, you can search the web for information.
Try to craft your response in markdown format, using appropriate headings, lists, and links to enhance readability.
Be detailed and thorough in your responses, ensuring that the user receives comprehensive information.
`;

@Injectable()
export class ChatService {
    graph;
    async compileChatGraph() {
        const model = new ChatGoogleGenerativeAI({
            model: 'gemini-2.0-flash-001'
        });

        const TavilySearchTool = new TavilySearch({
            maxResults: 3
        });

        const tools = [TavilySearchTool];

        const GraphState = Annotation.Root({
            messages: Annotation({
                reducer: addMessages,
            }),
        });

        const modelWithTools = model.bindTools(tools);

        const toolNode = new ToolNode(tools);

        const agent = async ({ messages }) => {
            if (messages.length === 1) {
                // Add the system message at the start
                messages.unshift(new SystemMessage(SystemPrompt));
            } else if (messages.length > 1) {
                // Replace the first message with the system message
                messages[0] = new SystemMessage(SystemPrompt);
            }
            console.log(messages);
            const response = await modelWithTools.invoke(messages);
            return {
                messages: [response]
            }
        }

        const workflow = new StateGraph(GraphState)
            .addNode("agent", agent)
            .addNode("tools", toolNode)
            .addEdge(START, "agent")
            .addConditionalEdges("agent", toolsCondition)
            .addEdge("tools", "agent")

        this.graph = workflow.compile({
            checkpointer,
        });
        console.log("Chat graph compiled successfully");
    }
    async getChatGraph() {
        if (!this.graph) {
            await this.compileChatGraph();
        }
        return this.graph;
    }
    async getMessages(threadId: string) {
        const readConfig = {
            configurable: { thread_id: threadId },
        };
        const response = await checkpointer.get(readConfig);
        if (!response) {
            return [];
        }
        const messages: { role: string, content: unknown, id: string }[] = [];
        (response?.channel_values?.messages as Array<object> || []).forEach((msg: any) => {
            if (msg.lc_id.includes("HumanMessage")) {
                messages.push({
                    role: "user",
                    content: msg.content,
                    id: msg.id,
                });
            } else if (msg.lc_id.includes("AIMessageChunk")) {
                messages.push({
                    role: "assistant",
                    content: msg.content,
                    id: msg.id,
                });
            }
        })
        return messages || [];
    }
}
