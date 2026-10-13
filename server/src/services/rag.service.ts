import path from "node:path";
import { fileURLToPath } from "node:url";
import { MongoClient } from "mongodb";
import { createAgent, tool } from "langchain";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { z } from "zod";

// ============================================================
// PATH CONFIGURATION
// ============================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================
// MONGODB CLIENT
// ============================================================

let mongoClient: MongoClient | null = null;

const getMongoClient = async (): Promise<MongoClient> => {
  if (!mongoClient) {
    const mongoUri = process.env.MONGODB_URI;

    if (!mongoUri) {
      throw new Error("MONGODB_URI is not set in .env");
    }

    mongoClient = new MongoClient(mongoUri);
    await mongoClient.connect();

    console.log(" MongoDB connected");
  }

  return mongoClient;
};

// ============================================================
// GOOGLE GEMINI EMBEDDINGS
// ============================================================

// gemini-embedding-001
// Actual embedding dimension is checked during initialization.

const getEmbeddings = () => {
  const apiKey = process.env.GOOGLE_API_KEY;

  if (!apiKey) {
    throw new Error("GOOGLE_API_KEY is not set in .env");
  }

  return new GoogleGenerativeAIEmbeddings({
    apiKey,
    model: "gemini-embedding-001",
  });
};

// ============================================================
// VECTOR STORE
// ============================================================

const getVectorStore = async () => {
  const client = await getMongoClient();

  const collection = client
    .db("neuroforge_db")
    .collection("knowledge_docs");

  return new MongoDBAtlasVectorSearch(getEmbeddings(), {
    collection: collection as any,
    indexName: "vector_index",
    textKey: "text",
    embeddingKey: "embedding",
  });
};

// ============================================================
// INITIALIZE KNOWLEDGE BASE
// ============================================================
// Runs once when the server starts.
//
// Flow:
// Knowledge file
//      ↓
// TextLoader
//      ↓
// Text splitting
//      ↓
// Gemini embeddings
//      ↓
// MongoDB Atlas
//      ↓
// knowledge_docs collection
// ============================================================

export const initializeKnowledgeBase = async (): Promise<void> => {
  const client = await getMongoClient();

  const collection = client
    .db("neuroforge_db")
    .collection("knowledge_docs");

  // ----------------------------------------------------------
  // CHECK WHETHER KNOWLEDGE BASE IS ALREADY INDEXED
  // ----------------------------------------------------------

  const docWithEmbedding = await collection.findOne({
    embedding: {
      $exists: true,
      $not: { $size: 0 },
    },
  });

  if (docWithEmbedding) {
    const count = await collection.countDocuments();

    console.log(
      ` Knowledge base ready (${count} chunks with embeddings)`
    );

    return;
  }

  // ----------------------------------------------------------
  // CLEAN OLD/EMPTY DOCUMENTS
  // ----------------------------------------------------------

  const existingCount = await collection.countDocuments();

  if (existingCount > 0) {
    console.log(
      ` Found ${existingCount} chunks with EMPTY embeddings — deleting & re-indexing...`
    );

    await collection.deleteMany({});
  }

  console.log(" Indexing NeuroForge knowledge base...");

  // ----------------------------------------------------------
  // INITIALIZE EMBEDDINGS
  // ----------------------------------------------------------

  const embeddings = getEmbeddings();

  // ----------------------------------------------------------
  // TEST GOOGLE API KEY + EMBEDDING MODEL
  // ----------------------------------------------------------

  try {
    const testResult = await embeddings.embedQuery("test");

    console.log(
      ` API key OK — embedding dimensions: ${testResult.length}`
    );
  } catch (error: any) {
    console.error(" Embedding test failed!");
    console.error(" Error:", error?.message || error);
    console.error(
      " Get your Google API key from: https://aistudio.google.com/apikey"
    );

    throw error;
  }

  // ----------------------------------------------------------
  // LOAD KNOWLEDGE FILE
  // ----------------------------------------------------------

  const filePath = path.join(
    __dirname,
    "../../knowledge-based/neuroforge-knowledge.txt"
  );

  console.log(` Loading knowledge file: ${filePath}`);

  const loader = new TextLoader(filePath);

  const docs = await loader.load();

  if (docs.length === 0) {
    throw new Error(
      "No documents found in NeuroForge knowledge base file"
    );
  }

  const totalCharacters = docs.reduce(
    (sum, doc) => sum + doc.pageContent.length,
    0
  );

  console.log(
    ` Loaded ${totalCharacters} characters from knowledge base`
  );

  // ----------------------------------------------------------
  // SPLIT DOCUMENT INTO CHUNKS
  // ----------------------------------------------------------

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await splitter.splitDocuments(docs);

  console.log(
    ` Split knowledge base into ${allSplits.length} chunks`
  );

  // ----------------------------------------------------------
  // CREATE VECTOR STORE
  // ----------------------------------------------------------

  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection: collection as any,
    indexName: "neuroforge_vector_index",
    textKey: "text",
    embeddingKey: "embedding",
  });

  // ----------------------------------------------------------
  // GENERATE EMBEDDINGS + STORE IN MONGODB
  // ----------------------------------------------------------

  console.log(" Generating embeddings and storing documents...");

  await vectorStore.addDocuments(allSplits);

  // ----------------------------------------------------------
  // VERIFY EMBEDDINGS
  // ----------------------------------------------------------

  const verifyDoc = await collection.findOne({
    embedding: {
      $exists: true,
      $not: { $size: 0 },
    },
  });

  if (
    verifyDoc &&
    Array.isArray(verifyDoc.embedding) &&
    verifyDoc.embedding.length > 0
  ) {
    console.log(
      ` ${allSplits.length} chunks stored (${verifyDoc.embedding.length}D embeddings)`
    );

    console.log(
      ` IMPORTANT: Create Atlas Vector Search index with numDimensions: ${verifyDoc.embedding.length}`
    );
  } else {
    await collection.deleteMany({});

    throw new Error(
      "Embeddings are empty! Google API returned no vectors."
    );
  }
};

// ============================================================
// RETRIEVAL TOOL
// ============================================================

const createRetrieveTool = (
  vectorStore: MongoDBAtlasVectorSearch
) => {
  return tool(
    async ({ query }: { query: string }) => {
      const retrievedDocs = await vectorStore.similaritySearch(
        query,
        3
      );

      return retrievedDocs
        .map(
          (doc) =>
            `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
        )
        .join("\n\n");
    },
    {
      name: "retrieve",

      description:
        "Retrieve information from the NeuroForge College knowledge base. " +
        "Use this for questions about courses, fees, admissions, mentors, campus, placements, and other college information.",

      schema: z.object({
        query: z.string(),
      }),
    }
  );
};

// ============================================================
// RAG RESPONSE
// ============================================================

export const getRAGResponse = async (
  question: string
): Promise<string> => {
  try {
    // --------------------------------------------------------
    // GET VECTOR STORE
    // --------------------------------------------------------

    const vectorStore = await getVectorStore();

    // --------------------------------------------------------
    // CREATE RETRIEVAL TOOL
    // --------------------------------------------------------

    const retrieve = createRetrieveTool(vectorStore);

    // --------------------------------------------------------
    // GEMINI CHAT MODEL
    // --------------------------------------------------------

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash",
      temperature: 0.7,
    });

    // --------------------------------------------------------
    // CREATE AGENT
    // --------------------------------------------------------

    const agent = createAgent({
      model,
      tools: [retrieve],

      systemPrompt:
        "You are NeuroForge Bot, a helpful AI counselor for NeuroForge College. " +
        "ALWAYS use the retrieve tool to search the knowledge base before answering. " +
        "Answer using information retrieved from the NeuroForge knowledge base. " +
        "Be concise, friendly, and professional. " +
        "Do not invent information that is not present in the knowledge base. " +
        "If the information is not found, say: " +
        "'I don't have that information right now. Click Talk to Us to speak with a counselor.'",
    });

    // --------------------------------------------------------
    // SEND USER QUESTION TO AGENT
    // --------------------------------------------------------

    const result = await agent.invoke({
      messages: [
        {
          role: "user",
          content: question,
        },
      ],
    });

    // --------------------------------------------------------
    // GET FINAL AGENT MESSAGE
    // --------------------------------------------------------

    const messages = result.messages;

    const lastMessage = messages[messages.length - 1];

    if (!lastMessage) {
      return "I couldn't generate a response. Please try again.";
    }

    return typeof lastMessage.content === "string"
      ? lastMessage.content
      : JSON.stringify(lastMessage.content);
  } catch (error) {
    console.error(" RAG Agent Error:", error);

    return "I'm having trouble right now. Please try again or click 'Talk to Us'.";
  }
  
};