// A small generator with a seed, so that a slot can be repeated. JavaScript cannot reproduce NumPy's
// generator, so the tests compare fixed inputs and statistics with Python, not samples.

export class Random {
  constructor(seed) {
    this.state = seed >>> 0;
    this.spare = null;
  }

  // Uniform in [0, 1), from the mulberry32 generator: 32 bits of state, enough for a demo.
  uniform() {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Standard normal, from the Box-Muller transform, which gives two values at a time.
  normal() {
    if (this.spare !== null) {
      const value = this.spare;
      this.spare = null;
      return value;
    }
    const radius = Math.sqrt(-2 * Math.log(1 - this.uniform()));
    const angle = 2 * Math.PI * this.uniform();
    this.spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  }
}
