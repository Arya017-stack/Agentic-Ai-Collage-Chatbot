import path from "node:path";
import { fileURLToPath } from "node:url";
import { MongoClient } from "mongodb";
import { createAgent, tool } from "langchain";
import { 
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { z } from "zod";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mongoClient: MongoClient | null = null;

const getMongoClient = async (): Promise<MongoClient> => {
  if (!mongoClient) {
    mongoClient = new MongoClient(process.env.MONGODB_URI || "");
    await mongoClient.connect();
  }
  return mongoClient;
};