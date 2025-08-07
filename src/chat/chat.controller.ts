import { BadRequestException, Controller, Get, Post, Query, Req, Res } from '@nestjs/common';
import type { Request, Response } from 'express';
import { PassThrough } from 'stream';
import { ChatService } from './chat.service';
import { HumanMessage } from '@langchain/core/messages';
import { v4 as uuidv4 } from 'uuid';

@Controller('chat')
export class ChatController {
    constructor(private readonly chatService: ChatService) {
        this.chatService.compileChatGraph();
    }

    @Post('/stream')
    async streamChat(@Req() req: Request, @Res() res: Response) {
        const { message, threadId } = req.body;

        const isNewThread = !threadId || threadId === '';

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();

        const stream = new PassThrough();
        stream.pipe(res);

        const graph = await this.chatService.getChatGraph();

        // Start LangGraph with input
        const updatedThreadId = isNewThread ? uuidv4() : threadId;
        const input = { messages: [new HumanMessage(message)] };
        const streamEvents = await graph.streamEvents(input, { configurable: { thread_id: updatedThreadId }, version: "v2" });

        try {
            stream.write(`event: thread_id\ndata: ${updatedThreadId}\n`);
            for await (const chunk of streamEvents) {
                if (chunk.event === 'on_chat_model_stream') {
                    if (chunk.data && chunk.data.chunk) {
                        stream.write(`${chunk.data.chunk.content}\n`);
                    }
                } else if (chunk.event === 'on_tool_end') {
                    if (chunk.name === 'TavilySearch' && chunk.data && chunk.data.output && chunk.data.output.content) {
                        stream.write(`event: web_search\ndata: ${chunk.data.output.content}\n`);
                    }
                }
            }
        } catch (err) {
            stream.write(`event: error\ndata: ${JSON.stringify(err)}\n`);
        } finally {
            stream.end();
        }
    }

    @Get('/messages')
    async getMessages(@Query('threadId') threadId: string) {
        if (!threadId) {
            throw new BadRequestException('threadId is required');
        }
        return await this.chatService.getMessages(threadId);
    }
}
