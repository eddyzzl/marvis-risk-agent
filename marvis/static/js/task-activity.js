// A busy indicator is a projection of live operations, never a mutable flag.
// Only the lease returned by claim can release that operation. Navigation does
// not cancel mutations: their owner persists until their own finally settles.
export function createTaskActivityOwner() {
  const operations = new Map();
  let sequence = 0;

  function claim(taskId, action, operation = action) {
    if (!action) throw new TypeError("An activity requires an action");
    const key = JSON.stringify([taskId || "", operation]);
    if (operations.has(key)) return null;
    const lease = Object.freeze({ taskId: taskId || null, action, key, id: ++sequence });
    operations.set(key, lease);
    return lease;
  }

  function release(lease) {
    if (!lease || operations.get(lease.key) !== lease) return false;
    operations.delete(lease.key);
    return true;
  }

  function action(taskId) {
    const owned = [...operations.values()].filter(item => item.taskId === (taskId || null));
    return owned.at(-1)?.action || null;
  }

  // Existing controllers expose balanced active/inactive callbacks. Give each
  // controller its own channel; a callback cannot release another controller's
  // operation, even when both render the same action label.
  function channel(actionId, onChange = () => {}) {
    const channelId = ++sequence;
    const leases = new Map();
    return (active, taskId) => {
      const queue = leases.get(taskId) || [];
      if (active) {
        const lease = claim(taskId, actionId, `channel:${channelId}:${++sequence}`);
        queue.push(lease);
        leases.set(taskId, queue);
        onChange(lease, true);
      } else if (queue.length) {
        const lease = queue.shift();
        if (!queue.length) leases.delete(taskId);
        if (release(lease)) onChange(lease, false);
      }
    };
  }

  return { claim, release, action, channel };
}
