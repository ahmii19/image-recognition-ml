import test from "node:test";
import assert from "node:assert/strict";

function computeBoundingBoxStyles(normalized) {
  const [ymin, xmin, ymax, xmax] = normalized;
  return {
    top: `${Number((ymin * 100).toFixed(4))}%`,
    left: `${Number((xmin * 100).toFixed(4))}%`,
    width: `${Number(((xmax - xmin) * 100).toFixed(4))}%`,
    height: `${Number(((ymax - ymin) * 100).toFixed(4))}%`,
  };
}

test("BoundingBox Math: normalized coordinates calculation is exact", () => {
  // [ymin, xmin, ymax, xmax]
  const normalized = [0.1, 0.2, 0.8, 0.7];
  const styles = computeBoundingBoxStyles(normalized);

  assert.equal(styles.top, "10%");
  assert.equal(styles.left, "20%");
  assert.equal(styles.width, "50%"); // 0.7 - 0.2 = 0.5 = 50%
  assert.equal(styles.height, "70%"); // 0.8 - 0.1 = 0.7 = 70%
});

test("BoundingBox Math: full canvas box scales to 100%", () => {
  const normalized = [0.0, 0.0, 1.0, 1.0];
  const styles = computeBoundingBoxStyles(normalized);

  assert.equal(styles.top, "0%");
  assert.equal(styles.left, "0%");
  assert.equal(styles.width, "100%");
  assert.equal(styles.height, "100%");
});
