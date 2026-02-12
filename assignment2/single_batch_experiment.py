import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from no_kv import Engine as NoKVEngine
from single_batch import Engine as KVEngine

INPUT_STR = "The University of Washington (UW) is a large, research-intensive public university system rooted in the Pacific Northwest and best known for the flagship Seattle campus, where ivy-covered brick buildings, modern labs, and wide lawns sit alongside cherry trees and views of Mount Rainier on clear days, creating a setting that feels simultaneously classic and distinctly Northwest. UW’s identity is shaped by its location in a fast-growing global city: Seattle’s mix of technology, healthcare, arts, maritime industry, and environmental innovation forms a living extension of the university, giving students and researchers daily proximity to internships, clinical training, policy work, startups, museums, and community organizations. Academically, UW spans the full breadth of disciplines—engineering and computer science, medicine and public health, natural sciences, social sciences, humanities, education, business, law, and the arts—organized into schools and colleges that support everything from broad liberal education to highly specialized professional pathways, with opportunities to combine fields through minors, certificates, interdisciplinary majors, research programs, and experiential learning. Its culture is strongly oriented toward discovery and impact: you’ll find undergraduates in research labs early in their college careers, graduate students pushing the boundaries of knowledge, and faculty leading collaborations that connect fundamental theory to real-world systems—whether that means building better medical devices, improving public health outcomes, advancing machine learning and data science, exploring climate and ocean dynamics, designing more sustainable infrastructure, or examining the social and ethical consequences of new technologies. UW’s scale is part of its power: a vast course catalog, an enormous range of student organizations, and extensive campus resources allow people with many different goals to find a niche, while also making the university feel like a city of its own, complete with libraries, galleries, performance spaces, recreation centers, dining halls, makerspaces, advising hubs, and cultural centers that support students’ academic and personal lives. The Seattle campus is often the image people carry when they think of UW—the iconic quad, the old trees, the historic architecture, and the energetic movement of students between classrooms and cafés—but UW is also a multi-campus system, with UW Bothell and UW Tacoma offering their own distinctive experiences, academic programs, and community connections, expanding access and shaping regional impact across Washington State. In everyday life, UW is defined by a rhythm that blends intensity and exploration: busy weeks of lectures, labs, problem sets, studio critiques, and group projects; late nights in libraries; conversations that continue long after class in dorm lounges and campus coffee shops; and weekends that might include hiking in the Cascades, exploring Seattle neighborhoods, visiting art and science museums, attending concerts, or gathering by Lake Washington and Lake Union, where the landscape constantly reminds you how close you are to water, mountains, and forests. This environment influences the university’s character—students tend to be practical, curious, and collaborative, often drawn to UW because they want rigorous training paired with meaningful application, and because the region’s industries and civic life reward people who can communicate across disciplines, learn quickly, and build things that work in the real world. UW’s research strengths are reinforced by major partnerships and institutional ecosystems: a strong medical and health sciences presence, including clinical training and biomedical research; deep connections to computing, engineering, and entrepreneurship in a region that has become a global hub for innovation; and longstanding leadership in fields like environmental science, oceanography, global health, education, and public policy. For many students, a defining feature of UW is the feeling that big ideas are not abstract—they’re visible in the labs and studios, in seminars and capstone projects, and in the way campus conversations often spill into broader questions about society, equity, sustainability, and the responsibilities that come with expertise. The university also carries traditions and spirit that anchor its modern ambition: “Husky” pride shows up in purple and gold everywhere, from orientation events to campus stores to the roar of fans at athletic competitions; game days bring a festival atmosphere that can unify students, alumni, and neighbors; and long-standing rituals—whether formal ceremonies or informal seasonal celebrations—help people feel part of a larger story. Athletics, while not the central reason many attend, contribute to community life through shared experiences and a sense of belonging, and the broader campus supports wellness and balance through recreation programs, intramural sports, mental health resources, and outdoor activities that take advantage of the region’s natural beauty. UW’s diversity is another cornerstone: students come from across Washington, across the United States, and from around the world, creating a community where many languages are spoken and where cultural exchange happens naturally through friendships, student groups, and campus events; for international students especially, UW can be a place to build global networks and gain confidence navigating new academic and professional environments. The arts and humanities are woven into this ecosystem, offering spaces for reflection and creative expression alongside technical training—performances, exhibitions, creative writing, philosophy, history, and languages enrich the campus and remind students that innovation is not only about new devices or algorithms but also about how people make meaning, build communities, and tell stories. UW’s libraries and learning spaces are emblematic of this blend of old and new: quiet study areas coexist with technology-enabled collaboration rooms, archival collections coexist with digital resources, and students move between solitary focus and teamwork depending on the demands of their fields. The university’s size also means navigating it requires initiative—students learn to advocate for themselves, seek out advising, join communities, and pursue opportunities early—yet for those who do, the payoff is the ability to tailor an education that matches their ambitions, whether that’s preparing for graduate school, launching a startup, entering industry, pursuing public service, becoming a clinician, or exploring a creative career. Career development is strengthened by the surrounding region and by UW’s alumni network, which extends into technology, medicine, education, government, nonprofits, and the arts; mentorship, internships, research experiences, and project-based learning can all become stepping stones to life after graduation. Importantly, UW is not just an academic brand or a set of buildings; it’s a living community with debates, challenges, and continual change—students negotiate workload and competition, learn to collaborate across differences, and engage with social issues that matter on campus and beyond. That engagement can take many forms: participating in service learning, working in labs that address global problems, contributing to student government, supporting cultural organizations, joining activism and advocacy efforts, or simply learning how to listen and communicate in a diverse environment. At its best, UW fosters a particular kind of confidence: not the confidence of having all the answers, but the confidence to ask better questions, to test ideas carefully, to work with others, and to keep learning when the world shifts. Whether you see UW as a gateway to a profession, a launchpad for research and innovation, a place to grow intellectually and personally, or a community that connects you to the Pacific Northwest and the wider world, the university’s defining promise is opportunity at scale—opportunity to study with experts, to explore across disciplines, to contribute to meaningful work, and to become part of a tradition that values knowledge not as an end in itself but as a tool for building healthier, more just, and more sustainable futures."


def time_generation(
    input_string: str, input_len: int, output_lens: list[int], use_kv_cache: bool
):
    if use_kv_cache:
        engine = KVEngine()
    else:
        engine = NoKVEngine()

    input_ids = engine.tokenizer.encode(input_string)
    if len(input_ids) > input_len:
        input_ids = input_ids[:input_len]
    pad_token = engine.tokenizer.pad_token_id or 0
    input_ids += [pad_token] * (input_len - len(input_ids))
    num_rounds = max(output_lens)
    print(f"{pad_token=}, {len(input_ids)=}, {num_rounds=}")
    return engine.timed_generate(
        input_ids, num_rounds=num_rounds, time_rounds=output_lens
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-min", type=int, default=128)
    parser.add_argument("--output-max", type=int, default=2048)
    parser.add_argument("--output-step", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=Path("results/single_batch"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    output_lens = list(range(args.output_min, args.output_max + 1, args.output_step))
    no_kv_times_cuda, no_kv_times_wall = time_generation(
        INPUT_STR, args.input_len, output_lens, use_kv_cache=False
    )
    kv_times_cuda, kv_times_wall = time_generation(
        INPUT_STR, args.input_len, output_lens, use_kv_cache=True
    )
    for output_len in output_lens:
        rows.append(
            {
                "output_len": output_len,
                "no_kv_ms": no_kv_times_cuda[output_len],
                "kv_ms": kv_times_cuda[output_len],
            }
        )
        print(
            f"output_len={output_len:4d} | "
            f"no_kv_time_cuda={no_kv_times_cuda[output_len]:.2f}ms | "
            f"kv_time_cuda={kv_times_cuda[output_len]:.2f}ms | "
            f"no_kv_time_wall={no_kv_times_wall[output_len]:.2f}ms | "
            f"kv_time_wall={kv_times_wall[output_len]:.2f}ms"
        )

    csv_path = args.out_dir / "single_batch_times.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["output_len", "no_kv_ms", "kv_ms"])
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["output_len"] for r in rows],
        [r["no_kv_ms"] for r in rows],
        marker="o",
        label="No KV cache",
    )
    plt.plot(
        [r["output_len"] for r in rows],
        [r["kv_ms"] for r in rows],
        marker="o",
        label="KV cache",
    )
    plt.xlabel("Output length (tokens)")
    plt.ylabel("Generation time (ms)")
    plt.title("Single-batch generation time vs output length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = args.out_dir / "single_batch_generation_time.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
