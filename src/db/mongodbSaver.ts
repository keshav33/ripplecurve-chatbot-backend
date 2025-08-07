import { MongoClient } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";

const client = new MongoClient("mongodb+srv://keshav33:keshav33@cluster0.ls4dhl8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0");
export const checkpointer = new MongoDBSaver({ client });