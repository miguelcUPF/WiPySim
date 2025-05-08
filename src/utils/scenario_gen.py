import random


def generate_random_network_deployment(
    sim_time_us: int,
    num_bss: int = 6,
    num_rl_bss: int = 1,
    disable_start_end: bool = False,
    seed=None,
):
    rng = random.Random(seed)

    rl_bss_ids = range(1, num_rl_bss + 1)
    bsss = []
    ap_id, sta_id = 1, 101

    for bss_id in range(1, num_bss + 1):
        is_rl = bss_id in rl_bss_ids

        traffic_model = rng.choice(["Poisson", "Bursty", "VR"])
        traffic = {
            "name": traffic_model,
            "traffic_load_kbps": rng.uniform(10_000, 500_000),
        }

        if not disable_start_end:
            traffic.update(
                {
                    "start_time_us": rng.randint(
                        0, int(0.25 * sim_time_us)
                    ),  # 0 to 1/4th of sim time
                    "end_time_us": rng.randint(
                        int(0.75 * sim_time_us), sim_time_us
                    ),  # 3/4th to end of sim time
                }
            )

        if traffic_model == "Bursty":
            traffic.update(
                {
                    "burst_size_pkts": rng.randint(10, 40),
                    "avg_inter_packet_time_us": rng.randint(2, 10),
                }
            )
        elif traffic_model == "VR":
            traffic.update(
                {
                    "fps": rng.choice([60, 90, 120]),
                    "avg_inter_packet_time_us": rng.randint(5, 10),
                }
            )

        bsss.append(
            {
                "id": bss_id,
                "ap": {
                    "id": ap_id,
                    "rl_driven": is_rl,
                },
                "stas": [{"id": sta_id}],
                "traffic_flows": [
                    {
                        "destination": sta_id,
                        "model": traffic,
                    }
                ],
            }
        )

        ap_id += 1
        sta_id += 1

    return bsss


def generate_random_scenario(seed=None, num_bss=6, num_rl_bss=1, sim_time_us=None, disable_start_end=False):
    if num_bss <= 1:
        raise ValueError("num_bss must be > 1")
    if num_rl_bss < 0:
        raise ValueError("num_rl_bss must be >= 0")

    rng = random.Random(seed)

    if sim_time_us is None:
        sim_time_us = rng.randint(1_000_000, 10_000_000)  # 1s to 10s

    bsss = generate_random_network_deployment(
        sim_time_us=sim_time_us,
        num_bss=num_bss,
        num_rl_bss=num_rl_bss,
        disable_start_end=disable_start_end,
        seed=seed,
    )
    return {
        "bsss_advanced": bsss,
        "sim_time_us": sim_time_us,
        "seed": seed,
    }
