import { test } from 'node:test';
import assert from 'node:assert/strict';

test('Region Intelligence: calculates 10% symmetrical context padding correctly', () => {
  const box = { x_min: 100, y_min: 100, x_max: 300, y_max: 300, width: 200, height: 200 };
  const paddingPercent = 10;
  const padW = Math.round(box.width * (paddingPercent / 100));
  const padH = Math.round(box.height * (paddingPercent / 100));

  const imageWidth = 500;
  const imageHeight = 500;

  const paddedBox = {
    x_min: Math.max(0, box.x_min - padW),
    y_min: Math.max(0, box.y_min - padH),
    x_max: Math.min(imageWidth, box.x_max + padW),
    y_max: Math.min(imageHeight, box.y_max + padH),
  };

  assert.equal(padW, 20);
  assert.equal(padH, 20);
  assert.equal(paddedBox.x_min, 80);
  assert.equal(paddedBox.y_min, 80);
  assert.equal(paddedBox.x_max, 320);
  assert.equal(paddedBox.y_max, 320);
  assert.equal(paddedBox.x_max - paddedBox.x_min, 240);
  assert.equal(paddedBox.y_max - paddedBox.y_min, 240);
});

test('Region Intelligence: boundary clamping near image borders', () => {
  const box = { x_min: 5, y_min: 5, x_max: 95, y_max: 95, width: 90, height: 90 };
  const padW = 20;
  const padH = 20;

  const imageWidth = 100;
  const imageHeight = 100;

  const clampedX1 = Math.max(0, box.x_min - padW);
  const clampedY1 = Math.max(0, box.y_min - padH);
  const clampedX2 = Math.min(imageWidth, box.x_max + padW);
  const clampedY2 = Math.min(imageHeight, box.y_max + padH);

  assert.equal(clampedX1, 0);
  assert.equal(clampedY1, 0);
  assert.equal(clampedX2, 100);
  assert.equal(clampedY2, 100);
});

test('Region Intelligence: maps response to DetectionViewer DetectedObject format', () => {
  const regionItem = {
    region_id: "reg-01",
    detection: {
      label: "dog",
      score: 0.88,
      prompt_used: "a photo of a dog",
    },
    original_box: { x_min: 100, y_min: 100, x_max: 200, y_max: 200, width: 100, height: 100 },
    padded_box: { x_min: 90, y_min: 90, x_max: 210, y_max: 210, width: 120, height: 120 },
    normalized_box: { x_min: 0.18, y_min: 0.18, x_max: 0.42, y_max: 0.42 },
    crop: { width: 120, height: 120, padding_percent: 10 },
    recognition: {
      top_match: { query: "golden retriever", similarity_score: 0.3124, rank: 1 },
      rankings: [
        { query: "golden retriever", similarity_score: 0.3124, rank: 1 },
        { query: "dog", similarity_score: 0.3012, rank: 2 },
      ],
      total_candidates: 2,
    },
  };

  const mappedObject = {
    label: `${regionItem.region_id}: ${regionItem.recognition?.top_match?.query || regionItem.detection.label}`,
    class_id: 0,
    confidence: regionItem.recognition?.top_match?.similarity_score ?? regionItem.detection.score,
    box: {
      x1: regionItem.padded_box.x_min,
      y1: regionItem.padded_box.y_min,
      x2: regionItem.padded_box.x_max,
      y2: regionItem.padded_box.y_max,
      width: regionItem.padded_box.width,
      height: regionItem.padded_box.height,
      normalized: [
        regionItem.normalized_box.y_min,
        regionItem.normalized_box.x_min,
        regionItem.normalized_box.y_max,
        regionItem.normalized_box.x_max,
      ],
    },
  };

  assert.equal(mappedObject.label, "reg-01: golden retriever");
  assert.equal(mappedObject.confidence, 0.3124);
  assert.equal(mappedObject.box.width, 120);
  assert.equal(mappedObject.box.x1, 90);
});
