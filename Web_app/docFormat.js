export function clauseFormat(text, numbering) {
  // Split the contract into clauses and return the array.
  // Future work: Include the level numbering in the output array.
  // Future work: Determine a better way to remove small sentences.
  // Future work: Consider grouping child clauses with their parent clauses. This may cause issues with LLM token limits.
  // Future work: Consider differences between case sensitive numbering systems (e.g., "a.", "A.", "(b)", "(B)", etc.)

  const debug = false
  // 1. Split the contract into clauses.
  // Split the text into clause using new line or carrage return characters.
  const clauseSpilt = text
    .split(/\r?\n/)
    .map((sentence) => sentence.trim()) // Remove leading and trailing whitespace
    .filter((sentence) => sentence.length > 0) // Remove any empty strings
    .filter((sentence) => sentence.length > 60) // Remove sentences that are less than 60 characters long.
    .filter((sentence) => !/^[“”""‘’'].*?\bmeans?\b/.test(sentence)) // Remove definitions. This is a heuristic and may not work for all cases.

  // 2. Find the numbering of clauses in the contract.
  // Look for sentectences that start with number followed by a period, a letter in parentheses, or a number in parentheses.
  const clauseNumberingRegex =
    /^\d+\s|^\d+\.\s|^\d+\.\d+\s|^\(\d+\)\s|^\w\.\s|^\(\w\)\s|^\([ivx]+\)\s/gm
  const clauseMatches = text.match(clauseNumberingRegex)
  // Determine the numbering hierarchy of the clauses
  clauseMatches.forEach((clauseMatch) => {
    if (/^\d+\s/.test(clauseMatch)) {
      if (debug) console.log("Level 1") // Single number (e.g., "1", "2")
    } else if (/^\d+\.\s/.test(clauseMatch)) {
      if (debug) console.log("Level 1") // Single number with dot notation (e.g., "1.", "2.")
    } else if (/^\d+\.\d+\s/.test(clauseMatch)) {
      if (debug) console.log("Level 2") // Number with dot notation (e.g., "1.1", "2.3")
    } else if (/^\w\.\s/.test(clauseMatch)) {
      if (debug) console.log("Level 3") // Letter with parenthesis (e.g., "a.", "B.")
    } else if (/^\(\w\)\s/.test(clauseMatch)) {
      if (debug) console.log("Level 3") // Letter with parenthesis (e.g., "(a)", "(B)")
    } else if (/^\([ivx]+\)\s/.test(clauseMatch)) {
      if (debug) console.log("Level 4") // Roman numeral with parenthesis (e.g., "(i)", "(ii)")
    }
  })
  if (debug) console.log(clauseMatches)

  // 3. Return the array of clauses.
  if (numbering) {
    // Remove the numbering from the clauses.
    return clauseSpilt.map((sentence) =>
      sentence.replace(clauseNumberingRegex, "").trim()
    )
  } else {
    return clauseSpilt
  }
}
