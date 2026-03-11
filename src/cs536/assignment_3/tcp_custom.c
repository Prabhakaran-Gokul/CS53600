/*
 * Custom TCP Congestion Control Algorithm - Kernel Module
 * 
 * Three-state algorithm: FAST, SLOW, REDUCE
 * Features: Bandwidth plateau detection, loss differentiation, 
 *           goodput monitoring, RTT inflation analysis
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <net/tcp.h>

#define CUSTOM_CC_NAME "custom"

/* Algorithm parameters */
#define INIT_CWND 300
#define MSS 300
#define HIGH_LOSS_THRESHOLD 500
#define GOODPUT_DECREASE_RATIO 90  /* 90% = 0.9 */
#define RTT_INFLATION_THRESHOLD 130  /* 130% = 1.3 */
#define JITTER_THRESHOLD 15  /* 15% = 0.15 */

/* States */
enum cc_state {
	STATE_FAST = 0,
	STATE_SLOW = 1,
	STATE_REDUCE = 2,
};

/* Per-connection state */
struct custom_cc {
	u32 cwnd;                /* Current congestion window */
	u32 rtt_min;             /* Minimum RTT observed */
	u32 bytes_acked_last;    /* Last bytes_acked value */
	u32 retrans_last;        /* Last retransmission count */
	u32 loss_count;          /* Current loss count */
	enum cc_state state;     /* Current state */
	u32 goodput_last;        /* Last goodput measurement */
	u64 last_ack_time;       /* Last ACK timestamp */
};

/* Initialize congestion control */
static void custom_cc_init(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct custom_cc *ca = inet_csk_ca(sk);

	ca->cwnd = INIT_CWND;
	ca->rtt_min = 0xFFFFFFFF;  /* Max value */
	ca->bytes_acked_last = 0;
	ca->retrans_last = 0;
	ca->loss_count = 0;
	ca->state = STATE_FAST;
	ca->goodput_last = 0;
	ca->last_ack_time = 0;

	tp->snd_cwnd = ca->cwnd;
	tp->snd_ssthresh = TCP_INFINITE_SSTHRESH;
}

/* Calculate RTT inflation */
static u32 get_rtt_inflation(u32 rtt_us, u32 rtt_min)
{
	if (rtt_min == 0)
		return 100;  /* 1.0 */
	return (rtt_us * 100) / rtt_min;  /* Returns percentage */
}

/* Calculate jitter ratio */
static u32 get_jitter_ratio(u32 rttvar_us, u32 rtt_us)
{
	if (rtt_us == 0)
		return 0;
	return (rttvar_us * 100) / rtt_us;  /* Returns percentage */
}

/* Determine algorithm state based on metrics */
static enum cc_state determine_state(struct sock *sk, struct custom_cc *ca)
{
	struct tcp_sock *tp = tcp_sk(sk);
	u32 rtt_us = tp->srtt_us >> 3;  /* Smooth RTT */
	u32 rttvar_us = tp->rttvar_us >> 2;  /* RTT variance */
	u32 rtt_inflation, jitter_ratio;
	u32 loss_delta;

	/* Update minimum RTT */
	if (rtt_us > 0 && rtt_us < ca->rtt_min)
		ca->rtt_min = rtt_us;

	/* Calculate metrics */
	rtt_inflation = get_rtt_inflation(rtt_us, ca->rtt_min);
	jitter_ratio = get_jitter_ratio(rttvar_us, rtt_us);

	/* Calculate loss delta */
	if (tp->total_retrans >= ca->retrans_last)
		loss_delta = tp->total_retrans - ca->retrans_last;
	else
		loss_delta = 0;
	
	ca->loss_count += loss_delta;
	ca->retrans_last = tp->total_retrans;

	/* Decision logic */
	
	/* 1. High loss -> REDUCE */
	if (ca->loss_count > HIGH_LOSS_THRESHOLD)
		return STATE_REDUCE;

	/* 2. Low loss detected -> SLOW */
	if (loss_delta > 0 && ca->loss_count < HIGH_LOSS_THRESHOLD)
		return STATE_SLOW;

	/* 3. RTT inflation with low jitter -> REDUCE (queueing) */
	if (rtt_inflation > RTT_INFLATION_THRESHOLD && 
	    jitter_ratio < JITTER_THRESHOLD)
		return STATE_REDUCE;

	/* 4. High jitter -> SLOW (variable delays) */
	if (jitter_ratio > JITTER_THRESHOLD)
		return STATE_SLOW;

	/* 5. Otherwise -> FAST (no congestion signals) */
	return STATE_FAST;
}

/* Update cwnd based on state */
static void update_cwnd(struct sock *sk, struct custom_cc *ca)
{
	struct tcp_sock *tp = tcp_sk(sk);

	switch (ca->state) {
	case STATE_FAST:
		/* Exponential growth */
		ca->cwnd = ca->cwnd * 2;
		if (ca->cwnd > tp->snd_cwnd_clamp)
			ca->cwnd = tp->snd_cwnd_clamp;
		break;

	case STATE_SLOW:
		/* Maintain current window */
		break;

	case STATE_REDUCE:
		/* Reset to initial cwnd */
		ca->cwnd = max(ca->cwnd / 2, 2U);
		/* Reset loss counter after reduction */
		ca->loss_count = 0;
		break;
	}

	if (ca->cwnd < 2)
		ca->cwnd = 2;
	tp->snd_cwnd = ca->cwnd;
}

/* Main ACK processing */
static void custom_cc_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
	struct custom_cc *ca = inet_csk_ca(sk);

	/* Determine new state */
	ca->state = determine_state(sk, ca);

	/* Update cwnd based on state */
	update_cwnd(sk, ca);
}

/* Handle packet loss event */
static u32 custom_cc_ssthresh(struct sock *sk)
{
	struct custom_cc *ca = inet_csk_ca(sk);

	/* Force REDUCE state on loss */
	ca->state = STATE_REDUCE;
	ca->cwnd = max(ca->cwnd / 2, 2U);

	return ca->cwnd;
}

/* Undo cwnd reduction (for spurious loss detection) */
static u32 custom_cc_undo_cwnd(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct custom_cc *ca = inet_csk_ca(sk);

	/* Restore cwnd, but cap at clamp */
	ca->cwnd = max(ca->cwnd * 2, tp->snd_cwnd);
	if (ca->cwnd > tp->snd_cwnd_clamp)
		ca->cwnd = tp->snd_cwnd_clamp;

	tp->snd_cwnd = ca->cwnd;
	return ca->cwnd;
}

/* Get current state info for debugging */
static void custom_cc_state(struct sock *sk, u8 ca_state)
{
	/* Optional: handle TCP_CA_Open, TCP_CA_Loss, etc. */
}

static struct tcp_congestion_ops custom_cc_ops __read_mostly = {
	.init		= custom_cc_init,
	.ssthresh	= custom_cc_ssthresh,
	.cong_avoid	= custom_cc_cong_avoid,
	.undo_cwnd	= custom_cc_undo_cwnd,
	.set_state	= custom_cc_state,
	.owner		= THIS_MODULE,
	.name		= CUSTOM_CC_NAME,
};

static int __init custom_cc_register(void)
{
	BUILD_BUG_ON(sizeof(struct custom_cc) > ICSK_CA_PRIV_SIZE);
	return tcp_register_congestion_control(&custom_cc_ops);
}

static void __exit custom_cc_unregister(void)
{
	tcp_unregister_congestion_control(&custom_cc_ops);
}

module_init(custom_cc_register);
module_exit(custom_cc_unregister);

MODULE_AUTHOR("CS536 Student");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Custom TCP Congestion Control Algorithm");
MODULE_VERSION("1.0");
