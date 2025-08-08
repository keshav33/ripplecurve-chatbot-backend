import { BadRequestException, Controller, Get, Post, Put, Query, Req, Res, UseGuards } from '@nestjs/common';
import type { Request, Response } from 'express';
import { PassThrough } from 'stream';
import { ChatService } from './chat.service';
import { HumanMessage } from '@langchain/core/messages';
import { v4 as uuidv4 } from 'uuid';
import { ClerkAuthGuard } from 'src/guard/clerk-auth.guard';

@Controller('chat')
@UseGuards(ClerkAuthGuard)
export class ChatController {
    constructor(private readonly chatService: ChatService) {
        this.chatService.compileChatGraph();
    }

    @Post('/stream')
    async streamChat(@Req() req: Request, @Res() res: Response) {
        const { message, threadId, user, humanId, assistantId } = req.body;

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
        const title = await this.chatService.createOrUpdateUserThread({
            userId: user.id,
            email: user.email,
            name: user.name,
            threadId: updatedThreadId,
            isNewThread,
            message
        });
        await this.chatService.pushMessages({
            threadId: updatedThreadId,
            userId: user.id,
            chat: {
                id: humanId,
                type: "user",
                text: message,
                isWebSearch: false,
                urls: [],
            }
        })
        const input = { messages: [new HumanMessage(message)] };
        const streamEvents = await graph.streamEvents(input, { configurable: { thread_id: updatedThreadId }, version: "v2" });

        try {
            let assistantMessage = "";
            let isWebSearch = false;
            let urls = [];
            stream.write(`event: thread_id\ndata: ${updatedThreadId}\n`);
            for await (const chunk of streamEvents) {
                if (chunk.event === 'on_chat_model_stream') {
                    if (chunk.data && chunk.data.chunk && !chunk.tags.includes('internal_summary')) {
                        assistantMessage += chunk.data.chunk.content;
                        stream.write(`${chunk.data.chunk.content}\n`);
                    }
                } else if (chunk.event === 'on_tool_end') {
                    if (chunk.name === 'TavilySearch' && chunk.data && chunk.data.output && chunk.data.output.content) {
                        const resultJson = JSON.parse(chunk.data.output.content);
                        const urls = (resultJson?.results || []).map((result) => result.url);
                        stream.write(`event: web_search\ndata: ${chunk.data.output.content}\n`);
                    }
                }
            }
            if (title) {
                stream.write(`event: thread_title\ndata: ${title}\n`);
            }
            await this.chatService.pushMessages({
                threadId: updatedThreadId,
                userId: user.id,
                chat: {
                    id: assistantId,
                    type: "assistant",
                    text: assistantMessage,
                    isWebSearch,
                    urls,
                }
            })
        } catch (err) {
            stream.write(`event: error\ndata: ${JSON.stringify(err)}\n`);
        } finally {
            stream.end();
        }
    }

    @Get('/history')
    async getChatHistory(@Query('userId') userId: string) {
        if (!userId) {
            throw new BadRequestException('userId is required');
        }
        return await this.chatService.getChatHistory(userId);
    }

    @Get('/messages')
    async getMessages(@Query('threadId') threadId: string) {
        if (!threadId) {
            throw new BadRequestException('threadId is required');
        }
        return await this.chatService.getMessages(threadId);
    }

    @Put('/feedback')
    async updateFeedback(@Query('threadId') threadId: string, @Query('messageId') messageId: string, @Query('feedback') feedback: string) {
        if (!threadId || !messageId || !feedback) {
            throw new BadRequestException('threadId, messageId and feedback are required');
        }
        await this.chatService.updateFeedback(threadId, messageId, feedback);
        return "Ok"
    }
}
