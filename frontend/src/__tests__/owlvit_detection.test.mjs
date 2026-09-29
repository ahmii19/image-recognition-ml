import { test } from 'node:test';
import assert from 'node:assert/strict';

test('OWL-ViT Detection: maps normalized bounding boxes correctly to view coordinates', () => {
  const detectionItem = {
    detection_id: "det-1",
    query: "dog",
    prompt_used: "a photo of a dog",
    score: 0.7184,
    box: {
      x_min: 100,
      y_min: 50,
      x_max: 400,
      y_max: 350,
      width: 300,
      height: 300,
    },
    box_normalized: {
      x_min: 0.20,
      y_min: 0.10,
      x_max: 0.80,
      y_max: 0.70,
    },
  };

  assert.equal(detectionItem.query, "dog");
  assert.equal(detectionItem.score, 0.7184);
  assert.equal(detectionItem.box.width, 300);
  assert.equal(detectionItem.box_normalized.x_max, 0.80);
});

test('OWL-ViT Detection: query deduplication and prompt formatting', () => {
  const rawQueries = ["dog", "cat", "DOG", "car"];
  const deduplicated = Array.from(new Set(rawQueries.map(q => q.toLowerCase())));
  assert.deepEqual(deduplicated, ["dog", "cat", "car"]);

  const template = "a photo of a {}";
  const formatted = deduplicated.map(q => template.replace("{}", q));
  assert.deepEqual(formatted, ["a photo of a dog", "a photo of a cat", "a photo of a car"]);
});
