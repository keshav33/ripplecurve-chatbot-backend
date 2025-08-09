import { MongoClient } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
const dotenv = require("dotenv");
dotenv.config();

const client = new MongoClient(process.env.MONGO_URI!, {
    maxPoolSize: 10,
    minPoolSize: 1,
    maxIdleTimeMS: 30000,
    serverSelectionTimeoutMS: 30000,
});

let isConnected = false;

export const initMongo = async () => {
    if (!isConnected) {
        await client.connect();
        isConnected = true;
        console.log("✅ MongoDB connected");
    }
};

export const checkpointer = new MongoDBSaver({
    client,
    dbName: "ripplecurve",
    checkpointCollectionName: "checkpoints",
    checkpointWritesCollectionName: "checkpoint_writes",
});

export const getCollection = (collectionName: string) => {
    return client.db("ripplecurve").collection(collectionName);
};

export const MESSAGES_THREAD = "messages_thread";
export const USER_THREADS_COLLECTION = "user_threads";
export const CHECKPOINT_COLLECTION = "checkpoints";
export const CHECKPOINT_WRITES_COLLECTION = "checkpoint_writes";
