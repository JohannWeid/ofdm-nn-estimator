// The int8 correction network in plain JavaScript, without any library.
//
// A quantized value q stands for the real value (q - zero) * step. Writing a convolution
//     y = sum(w * x) + b
// in those terms gives for every output channel
//     q_y = zero_y + round((sum(q_w * (q_x - zero_x)) + q_b) * step_x * step_w / step_y)
// Everything is integer arithmetic except the last factor, which is float32 as in ONNX Runtime.
// tests/network.test.mjs checks that the result is bit-identical to ONNX Runtime. The parameters
// come from scripts/export_web_data.py.

const SYMBOLS = 14;
const SUBCARRIERS = 72;
const SLOT_SIZE = SYMBOLS * SUBCARRIERS;
export const INPUT_SIZE = 3 * SLOT_SIZE;
export const OUTPUT_SIZE = 2 * SLOT_SIZE;
// One row and one column of zeros around every channel: the 3x3 filters use padding 1.
const ROWS = SYMBOLS + 2;
const COLUMNS = SUBCARRIERS + 2;

const { fround } = Math;

// Rounds to the nearest integer and a half to the even neighbour, like the x86 conversion.
// Math.round alone would round every half up.
export function roundHalfEven(value) {
  const rounded = Math.round(value);
  return Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}

function clampToInt8(value) {
  return value < -128 ? -128 : value > 127 ? 127 : value;
}

function prepareLayer(layer, input) {
  const { in_channels: inChannels, out_channels: outChannels, output } = layer;
  return {
    inChannels,
    outChannels,
    inputZero: input.zero,
    outputZero: output.zero,
    weights: Int8Array.from(layer.weights),
    biases: Int32Array.from(layer.biases),
    // In this order and in float32, like ONNX Runtime.
    multipliers: Float32Array.from(layer.weight_steps, (step) =>
      fround(fround(input.step * step) / output.step),
    ),
    padded: new Int16Array(inChannels * ROWS * COLUMNS),
    sums: new Int32Array(SLOT_SIZE),
    output: new Int8Array(outChannels * SLOT_SIZE),
  };
}

// Takes the parsed JSON of a model and returns what runCorrection needs, with its working memory.
export function prepare(model) {
  const { layers } = model;
  if (layers[0].in_channels !== 3 || layers.at(-1).out_channels !== 2) {
    throw new Error("the network must map 3 input channels to 2 output channels");
  }
  let input = model.input;
  const prepared = layers.map((layer) => {
    const result = prepareLayer(layer, input);
    input = layer.output;
    return result;
  });
  return {
    input: model.input,
    output: layers.at(-1).output,
    layers: prepared,
    quantizedInput: new Int8Array(INPUT_SIZE),
  };
}

function convolve(layer, input) {
  const { inChannels, outChannels, inputZero, outputZero } = layer;
  const { weights, biases, multipliers, padded, sums, output } = layer;

  // The border stays zero: a value of zero after subtracting the zero point is a real zero.
  for (let c = 0; c < inChannels; c++) {
    for (let s = 0; s < SYMBOLS; s++) {
      const from = (c * SYMBOLS + s) * SUBCARRIERS;
      const to = (c * ROWS + s + 1) * COLUMNS + 1;
      for (let k = 0; k < SUBCARRIERS; k++) {
        padded[to + k] = input[from + k] - inputZero;
      }
    }
  }

  let w = 0; // the weights are stored in the order output channel, input channel, row, column
  for (let o = 0; o < outChannels; o++) {
    sums.fill(biases[o]);
    for (let c = 0; c < inChannels; c++) {
      for (let row = 0; row < 3; row++) {
        for (let column = 0; column < 3; column++) {
          const weight = weights[w++];
          if (weight === 0) {
            continue;
          }
          for (let s = 0; s < SYMBOLS; s++) {
            const from = (c * ROWS + s + row) * COLUMNS + column;
            const to = s * SUBCARRIERS;
            for (let k = 0; k < SUBCARRIERS; k++) {
              sums[to + k] += weight * padded[from + k];
            }
          }
        }
      }
    }
    const multiplier = multipliers[o];
    const base = o * SLOT_SIZE;
    for (let i = 0; i < SLOT_SIZE; i++) {
      const scaled = fround(fround(sums[i]) * multiplier);
      output[base + i] = clampToInt8(roundHalfEven(scaled) + outputZero);
    }
  }
  return output;
}

// input holds INPUT_SIZE values, correction has room for OUTPUT_SIZE values, both float32 in the
// order channel, symbol, subcarrier. The same layout as CnnCorrection in cpp/.
export function runCorrection(network, input, correction) {
  const { quantizedInput, layers } = network;
  const { step, zero } = network.input;
  for (let i = 0; i < INPUT_SIZE; i++) {
    quantizedInput[i] = clampToInt8(roundHalfEven(fround(input[i] / step)) + zero);
  }
  let activation = quantizedInput;
  for (const layer of layers) {
    activation = convolve(layer, activation);
  }
  const { step: outputStep, zero: outputZero } = network.output;
  for (let i = 0; i < OUTPUT_SIZE; i++) {
    correction[i] = fround((activation[i] - outputZero) * outputStep);
  }
}
