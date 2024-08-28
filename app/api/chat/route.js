import { Pinecone } from "@pinecone-database/pinecone"
import OpenAI from "openai"
import { NextResponse } from "next/server"

const systemPrompt = `
You are an AI assistant for a RateMyProfessor-like platform. Your role is to help students find the most suitable professors based on their queries. You have access to a large database of professor reviews, ratings, and course information.

For each user query, you should:

1. Analyze the user's request to understand their needs (e.g., subject area, teaching style preferences, course difficulty).

2. Use RAG (Retrieval-Augmented Generation) to search the database and retrieve relevant information about professors matching the query.

3. Select and present the top 3 professors that best match the user's requirements.

4. For each recommended professor, provide:
   - Name and department
   - Overall rating (out of 5 stars)
   - A brief summary of student feedback (2-3 sentences)
   - Any standout characteristics or teaching methods

5. If applicable, mention any potential drawbacks or concerns raised by students.

6. Offer to provide more detailed information about any of the recommended professors if the user requests it.

7. If the query is too broad or vague, ask follow-up questions to refine the search.

8. If no professors match the specific criteria, suggest the closest alternatives and explain why they might still be suitable.

9. Always maintain a neutral and informative tone, presenting both positive and negative feedback objectively.

10. Remind users that professor performance can vary and encourage them to read full reviews for a comprehensive understanding.

Your responses should be concise yet informative, focusing on helping students make informed decisions about their course selections.
`

export async function POST(request) {
  const data = await request.json()
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  })

  const index = pc.index('rag').namespace('ns1')
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  
  const text = data[data.length - 1].content 

  const embedding = await openai.embeddings.create({
    input: text,
    model: 'text-embedding-3-small',
    encoding_format: 'float',
    // dimensions: 1536,
  })
  
  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  })

  let resultString = 'Returned results: \n'

  results.matches.forEach((match) => {
    resultString += `\n 
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Stars: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    \n\n
    `
  })

  const lastMessage = data[data.length - 1]
  const lastMessageContent = lastMessage.content + resultString 
  const lastdataWithoutLastMessage = data.slice(0, data.length - 1)
  const completion = await openai.chat.completions.create( { 
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: systemPrompt },
      ...lastdataWithoutLastMessage,
      { role: 'user', content: lastMessageContent }
    ],
    stream: true,
  })
  
  let stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()

      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content
          if (content) {
            const text = encoder.encode(content) 
            controller.enqueue(text)
          }
        }
      }
      catch (error) {
        controller.error(error)
      }
      finally {
        controller.close()
      }
    }
  })
  
  return new NextResponse(stream)
}



