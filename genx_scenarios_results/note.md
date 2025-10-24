# Place genx simulations results here.

## Input data structure
Place each scenario's results in a separate folder within this folder. For example:
```
genx_scenarios_results/
    scenario_1/
        inputs/
            inputs_p1/
            inputs_p2/
            .
            .
            .
        results/
    scenario_2/
        inputs/
            inputs_p1/
            inputs_p2/
            .
            .
            .
        results/ 
    .
    .
    .
```

## Reformatting instructions

To reformat this data into plottable data,

```python
python reformat_data_for_plot/format_genx_result_for_plotting.py
```

(default input: (genx_scenarios_results); default output:(genx-scenarios))
