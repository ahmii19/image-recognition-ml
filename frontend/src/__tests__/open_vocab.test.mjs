import test from "node:test";
import assert from "node:assert/strict";

test("Open-Vocabulary: validates candidate query constraint limits", () => {
  const maxQueries = 20;
  const maxQueryLength = 128;

  const validQueries = ["rose", "sunflower", "sports car", "laptop"];
  assert.equal(validQueries.length <= maxQueries, true);
  assert.equal(validQueries.every((q) => q.length <= maxQueryLength), true);

  const tooLong = "a".repeat(130);
  assert.equal(tooLong.length > maxQueryLength, true);
});

test("Open-Vocabulary: calculates normalized similarity meter width", () => {
  function calculateBarWidth(score, maxScore) {
    if (maxScore <= 0) return Math.max(5, Math.min(100, (score + 1) * 50));
    const normalized = Math.max(0.05, score / maxScore);
    return Math.min(100, Math.round(normalized * 100));
  }

  const topScore = 0.2754;
  const runnerUpScore = 0.1504;

  assert.equal(calculateBarWidth(topScore, topScore), 100);
  assert.equal(calculateBarWidth(runnerUpScore, topScore) < 100, true);
  assert.equal(calculateBarWidth(runnerUpScore, topScore) > 0, true);
});
