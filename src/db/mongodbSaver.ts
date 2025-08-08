import { MongoClient } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";

const client = new MongoClient("mongodb+srv://keshav33:keshav33@cluster0.ls4dhl8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0");
export const checkpointer = new MongoDBSaver({ client, dbName: "ripplecurve", checkpointCollectionName: "checkpoints", checkpointWritesCollectionName: "checkpoint_writes" });

export const getCollection = (collectionName: string) => {
    return client.db("ripplecurve").collection(collectionName);
};

export const MESSAGES_THREAD = "messages_thread";
export const USER_THREADS_COLLECTION = "user_threads";
export const CHECKPOINT_COLLECTION = "checkpoints";
export const CHECKPOINT_WRITES_COLLECTION = "checkpoint_writes";
