/*
 * Custom TCP Congestion Control - eBPF Implementation
 * 
 * Requires Linux kernel 5.6+ with CONFIG_BPF_SYSCALL=y
 *
 * Note: this sock_ops example tracks congestion signals in BPF state, but it
 * does not directly program TCP cwnd. For active cwnd control in eBPF, use
 * the TCP struct_ops interface (BPF_PROG_TYPE_STRUCT_OPS).
 * Compile with: clang -O2 -target bpf -c tcp_custom_bpf.c -o tcp_custom_bpf.o
 */

#include <linux/bpf.h>
#include <linux/types.h>
#include <bpf/bpf_helpers.h>

/* BPF congestion control operations structure */
struct bpf_tcp_cc_ops {
	void (*init)(struct bpf_sock_ops *skops);
	void (*release)(struct bpf_sock_ops *skops);
	__u32 (*ssthresh)(struct bpf_sock_ops *skops);
	void (*cong_avoid)(struct bpf_sock_ops *skops, __u32 ack, __u32 acked);
	void (*set_state)(struct bpf_sock_ops *skops, __u8 new_state);
	void (*cwnd_event)(struct bpf_sock_ops *skops, __u32 event);
	__u32 (*undo_cwnd)(struct bpf_sock_ops *skops);
};

/* Algorithm parameters */
#define INIT_CWND 300
#define HIGH_LOSS_THRESHOLD 500
#define RTT_INFLATION_THRESHOLD 130  /* 130% = 1.3 */
#define JITTER_THRESHOLD 15  /* 15% = 0.15 */

/* States */
#define STATE_FAST 0
#define STATE_SLOW 1
#define STATE_REDUCE 2

/* Per-connection state stored in BPF map */
struct cc_state {
	__u32 cwnd;
	__u32 rtt_min;
	__u32 loss_count;
	__u32 retrans_last;
	__u8 state;
};

/* BPF map to store per-connection state */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10000);
	__type(key, __u64);  /* Socket cookie */
	__type(value, struct cc_state);
} cc_state_map SEC(".maps");

/* Helper to get/create state */
static struct cc_state *get_state(struct bpf_sock_ops *skops)
{
	__u64 cookie = bpf_get_socket_cookie(skops);
	struct cc_state *state = bpf_map_lookup_elem(&cc_state_map, &cookie);
	
	if (!state) {
		struct cc_state new_state = {
			.cwnd = INIT_CWND,
			.rtt_min = 0xFFFFFFFF,
			.loss_count = 0,
			.retrans_last = 0,
			.state = STATE_FAST,
		};
		bpf_map_update_elem(&cc_state_map, &cookie, &new_state, BPF_ANY);
		state = bpf_map_lookup_elem(&cc_state_map, &cookie);
	}
	
	return state;
}

/* Calculate RTT inflation */
static __u32 get_rtt_inflation(__u32 rtt_us, __u32 rtt_min)
{
	if (rtt_min == 0)
		return 100;
	return (rtt_us * 100) / rtt_min;
}

/* Calculate jitter ratio */
static __u32 get_jitter_ratio(__u32 rttvar_us, __u32 rtt_us)
{
	if (rtt_us == 0)
		return 0;
	return (rttvar_us * 100) / rtt_us;
}

/* Determine algorithm state */
static __u8 determine_state(struct bpf_sock_ops *skops, struct cc_state *state)
{
	__u32 rtt_us = skops->rtt_us;
	__u32 rttvar_us = skops->rttvar_us;
	__u32 rtt_inflation, jitter_ratio;
	__u32 total_retrans = skops->total_retrans;
	__u32 loss_delta;

	/* Update minimum RTT */
	if (rtt_us > 0 && rtt_us < state->rtt_min)
		state->rtt_min = rtt_us;

	/* Calculate metrics */
	rtt_inflation = get_rtt_inflation(rtt_us, state->rtt_min);
	jitter_ratio = get_jitter_ratio(rttvar_us, rtt_us);

	/* Calculate loss delta */
	if (total_retrans >= state->retrans_last)
		loss_delta = total_retrans - state->retrans_last;
	else
		loss_delta = 0;

	state->loss_count += loss_delta;
	state->retrans_last = total_retrans;

	/* Decision logic */
	if (state->loss_count > HIGH_LOSS_THRESHOLD)
		return STATE_REDUCE;

	if (loss_delta > 0 && state->loss_count < HIGH_LOSS_THRESHOLD)
		return STATE_SLOW;

	if (rtt_inflation > RTT_INFLATION_THRESHOLD && 
	    jitter_ratio < JITTER_THRESHOLD)
		return STATE_REDUCE;

	if (jitter_ratio > JITTER_THRESHOLD)
		return STATE_SLOW;

	return STATE_FAST;
}

/* Update cwnd based on state */
static void update_cwnd(struct bpf_sock_ops *skops, struct cc_state *state)
{
	(void)skops;

	switch (state->state) {
	case STATE_FAST:
		state->cwnd = state->cwnd * 2;
		if (state->cwnd > 65535)  /* Max cwnd */
			state->cwnd = 65535;
		break;

	case STATE_SLOW:
		/* Maintain current window */
		break;

	case STATE_REDUCE:
		state->cwnd = INIT_CWND;
		state->loss_count = 0;
		break;
	}
}

/* Main BPF program */
SEC("sockops")
int custom_cc_main(struct bpf_sock_ops *skops)
{
	struct cc_state *state;
	__u32 op = skops->op;

	/* Only handle congestion control operations */
	switch (op) {
	case BPF_SOCK_OPS_TCP_CONNECT_CB:
	case BPF_SOCK_OPS_ACTIVE_ESTABLISHED_CB:
	case BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB:
		/* Initialize state */
		get_state(skops);
		break;

	case BPF_SOCK_OPS_STATE_CB:
		/* Handle state changes */
		state = get_state(skops);
		if (state) {
			state->state = determine_state(skops, state);
			update_cwnd(skops, state);
		}
		break;

	case BPF_SOCK_OPS_RTT_CB:
		/* Update on RTT measurements */
		state = get_state(skops);
		if (state) {
			state->state = determine_state(skops, state);
			update_cwnd(skops, state);
		}
		break;
	}

	return 1;
}

char _license[] SEC("license") = "GPL";
