import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js"

let tokenizer, session

// Load the ONNX model and tokenizer
console.log("Loading ONNX Runtime WebAssembly backend...")
var downLoadingModel = true
tokenizer = await AutoTokenizer.from_pretrained("../web_app/model/")
session = await ort.InferenceSession.create(
  "./model/ziggy_model_quantized.onnx"
)
downLoadingModel = false
console.log("Model and tokenizer successfully loaded.")

// Softmax function
function softmax(logits) {
  const maxLogit = Math.max(...logits)
  const exps = logits.map((x) => Math.exp(x - maxLogit))
  const sumExps = exps.reduce((a, b) => a + b)
  return exps.map((value) => value / sumExps)
}

// Predict the sentence
export async function predictSentence(sentence) {
  try {
    async function replaceUnknownWords(sentence, encoding) {
      // console.log("Sentence: ", sentence)
      // console.log("Encoding: ", encoding)
      console.log("Tokenized: ", tokenizer.decode(encoding))
      // Check if the encoding contains unknown words
      for (let i = 0; i < encoding.length; i++) {
        if (encoding[i] === tokenizer.unk_token_id) {
          // Add the word from the sentence to the unknown words list
          const word = sentence.split(" ")[i]
          if (word !== undefined) {
            unknownWords.push(word)
          }
        }
      }
      return
    }
    // Tokenize the sentence
    sentence = sentence.toLowerCase()
    let encoding = tokenizer.encode(sentence)
    await replaceUnknownWords(sentence, encoding)
    if (encoding.length > tokenizer.model_max_length) {
      encoding = encoding[(0, tokenizer.model_max_length)]
    } else {
      encoding = encoding.concat(
        Array(tokenizer.model_max_length - encoding.length).fill(0)
      )
    }
    // Create the input tensors
    const inputIds = new ort.Tensor(
      "int64",
      BigInt64Array.from(encoding.map((id) => BigInt(id))),
      [1, encoding.length]
    )
    const attentionMask = new ort.Tensor(
      "float32", // Correct type for attention_mask
      Float32Array.from(encoding.map((mask) => (mask !== 0 ? 1.0 : 0.0))), // Convert to float
      [1, encoding.length]
    )
    // Prepare the input feeds
    const feeds = {
      input_ids: inputIds,
      attention_mask: attentionMask,
    }
    // Run the model
    const output = await session.run(feeds)
    const probabilities = softmax(output.logits.data)
    console.log("Probabilities: ", probabilities)
    // Create the output object with indexes and probabilities
    const outputObject = []
    for (let i = 0; i < probabilities.length; i++) {
      outputObject.push({
        index: i,
        probability: probabilities[i],
      })
    }
    console.log("Output object: ", outputObject)
    // Sort the probabilities from highest to lowest
    const sortedProbabilities = outputObject.sort(
      (a, b) => b.probability - a.probability
    )
    console.log("Sorted probabilities: ", sortedProbabilities)
    return sortedProbabilities
  } catch (e) {
    console.error(e)
    throw e
  }
}
