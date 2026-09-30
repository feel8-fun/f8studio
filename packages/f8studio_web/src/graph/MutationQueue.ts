/** Serialize mutations; a failed operation invalidates work queued against it. */
export class MutationQueue {
  private tail: Promise<void> = Promise.resolve();
  private generation = 0;
  private count = 0;

  get pending(): number { return this.count; }

  enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const generation = this.generation;
    this.count += 1;
    const result = this.tail.then(async () => {
      if (generation !== this.generation) throw new Error('An earlier edit failed. Review the current graph and retry this edit.');
      try {
        return await operation();
      } catch (error) {
        this.generation += 1;
        throw error;
      }
    });
    this.tail = result.then(
      () => { this.count -= 1; },
      () => { this.count -= 1; },
    );
    return result;
  }
}
