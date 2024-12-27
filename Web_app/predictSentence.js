import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js"

const labels = ["Action", "Description", "Dialogue"]
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

export async function predictSentence(sentence) {
  try {
    async function replaceUnknownWords(sentence, tokenizer) {
      const words = sentence.split(" ")
      const knownWords = []
      for (const word of words) {
        const encoding = await tokenizer.encode(word)
        // Check if the encoded output contains an unknown token ID
        if (encoding.includes(tokenizer.unk_token_id)) {
          knownWords.push("[UNK]")
          unknownWords.push(word)
        } else {
          knownWords.push(word)
        }
      }
      return knownWords.join(" ")
    }
    const cleanSentence = await replaceUnknownWords(sentence, tokenizer)
    let encoding = tokenizer.encode(cleanSentence)
    if (encoding.length > 512) {
      encoding = encoding[(0, 512)]
    } else {
      encoding = encoding.concat(Array(512 - encoding.length).fill(0))
    }

    const inputIds = new ort.Tensor(
      "int64",
      BigInt64Array.from(encoding.map((id) => BigInt(id))),
      [1, encoding.length]
    )
    const attentionMask = new ort.Tensor(
      "int64",
      BigInt64Array.from(encoding.map((mask) => BigInt(mask))),
      [1, encoding.length]
    )
    const feeds = {
      input_ids: inputIds,
      attention_mask: attentionMask,
    }
    const output = await session.run(feeds)
    const probabilities = softmax(output.logits.data)
    const labeledProbabilities = labels.map((label, i) => ({
      label: label,
      probability: probabilities[i],
    }))
    labeledProbabilities.sort((a, b) => b.probability - a.probability)
    // Return the labeled probabilities
    return labeledProbabilities
  } catch (e) {
    console.error(e)
    throw e
  }
}
