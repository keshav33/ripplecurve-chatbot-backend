import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common';
import { Request } from 'express';
import { verifyToken } from '@clerk/backend';
const dotenv = require("dotenv");
dotenv.config();

@Injectable()
export class ClerkAuthGuard implements CanActivate {
    async canActivate(context: ExecutionContext): Promise<boolean> {
        const request = context.switchToHttp().getRequest<Request>();
        const authHeader = request.headers.authorization;

        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            throw new UnauthorizedException('Missing or invalid Authorization header');
        }

        const token = authHeader.split(' ')[1];

        try {
            const { sub } = await verifyToken(token, { secretKey: process.env.CLERK_SECRET_KEY });
            request['userId'] = sub
            return true;
        } catch (error) {
            throw new UnauthorizedException('Invalid Clerk token');
        }
    }
}