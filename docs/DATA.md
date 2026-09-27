# Data

The bundled inputs are the 14 original AFIT files in `benchmarks/afit/`.

- Source: [AFSCN Scheduling Problems](https://www.cs.colostate.edu/sched/data.html).
- Archive: [AFIT.tgz](https://www.cs.colostate.edu/sched/data/AFIT.tgz).
- Original documentation: [UPSTREAM_README.txt](../benchmarks/afit/UPSTREAM_README.txt).
- File checksums: [source-manifest.json](../benchmarks/afit/source-manifest.json).

| Scenario | Low file | High file | Tasks | Antennas used |
|---|---|---|---:|---:|
| AFIT1 | reqlf12.dat | requp12.dat | 322 | 19 |
| AFIT2 | reqlf13.dat | requp13.dat | 302 | 19 |
| AFIT3 | reqlf14.dat | requp14.dat | 310 | 19 |
| AFIT4 | reqlf15.dat | requp15.dat | 318 | 19 |
| AFIT5 | reqlf16.dat | requp16.dat | 305 | 19 |
| AFIT6 | reqlf17.dat | requp17.dat | 299 | 19 |
| AFIT7 | reqlf18.dat | requp18.dat | 297 | 19 |

## Reader

Columns are request ID, antenna ID, visibility start, visibility end, duration, turnaround time, and optional orbit information. The reader ignores the orbit column, keeps low/high request IDs separate, and uses equal profits `p_i = 1` by default.

The model uses 19 antennas. Low-orbit requests use fixed windows; high-orbit start times use a one-unit grid from `ceil(start)` to `floor(end - duration)`. Turnaround precedes the later task and may begin before time zero. The original zero-duration request in AFIT3 is retained.

## Optional synthetic instances

The generator creates new synthetic instances in an external directory:

```bash
python -m nsga2_sats.generate_data --output ../nsga2-sats-large-inputs --seed 20260918
python run.py --profile large --data-dir ../nsga2-sats-large-inputs --dry-run
```

| Scenario | File pair | Tasks | Antennas |
|---|---|---:|---:|
| 8 | reqlf19.dat / requp19.dat | 500 | 24 |
| 9 | reqlf20.dat / requp20.dat | 750 | 30 |
| 10 | reqlf21.dat / requp21.dat | 1000 | 38 |

Generation samples AFIT1 request groups within orbit types, assigns synthetic antenna labels, and shifts windows within the scheduling horizon. Its manifest records the seed, settings, template hashes, and generated file hashes.
