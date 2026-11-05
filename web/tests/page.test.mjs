// The CI has no browser, so these tests read the files: mistakes that only a browser would notice.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const WEB = new URL("../", import.meta.url);
const read = (path) => readFileSync(new URL(path, WEB), "utf8");
const html = read("index.html");
const app = read("js/app.js");

test("every element the script looks up exists on the page", () => {
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
  const wanted = [...app.matchAll(/element\("([^"]+)"\)/g)].map((match) => match[1]);
  for (const name of ["ls", "cnn", "perfect"]) {
    wanted.push(`nmse-${name}`, `ber-${name}`, `constellation-${name}`, `constellation-${name}-note`);
  }
  for (const preset of ["training", "unseen", "fails"]) {
    wanted.push(`preset-${preset}`);
  }
  for (const map of ["map-truth", "map-ls", "map-cnn", "map-error-ls", "map-error-cnn"]) {
    wanted.push(map);
  }
  assert.ok(wanted.length > 30);
  for (const id of wanted) {
    assert.ok(ids.has(id), `missing element ${id}`);
  }
});

test("the page uses relative paths, because it does not live at the root of its domain", () => {
  for (const match of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    const url = match[1];
    assert.ok(url.startsWith("https://") || !url.startsWith("/"), `absolute path ${url}`);
  }
});

test("the page loads nothing from other servers except links", () => {
  assert.doesNotMatch(html, /<script[^>]+src="https?:/);
  assert.doesNotMatch(html, /<link[^>]+href="https?:/);
});
