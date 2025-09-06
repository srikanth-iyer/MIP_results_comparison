# CHANGELOG
scenarios:
- p1_High_Elect_Mid_RE
- p3_High_Elect_Low_RE
- p4_Mod_Elect_Low_RE
## For all scenarios (p1, p2 and p4)
- Run.jl file:
    - Changed the solver to Gurobi in the genx_settings.yml file.
    - Added run_genx_case!(dirname(@__FILE__),Gurobi.Optimizer) to the run.jl file.
- genx_settings.yml
    - ModelingToGenerateAlternatives = 1 in genx_settings.yml
- gurobi_settings.yml
    - BarObjRng commented out in the settings file.
- CO2_cap.csv
    - added a number prefix "_1" to co2_max_mton column in the CO2_cap.csv file in policies in all scenarios
        - save original file as CO2_cap_OG.csv
- added Resource_Type:
    - based on tech+cluster for Vre.csv 
        ```python
        # Combine 'technology' and 'cluster' into 'Resource_Type'
        df['Resource_Type'] = df['technology'].str.replace(' ', '_') + '_' + df['cluster'].astype(str)
        ```
    - based on technology for Thermal.csv, Must_run.csv, Hydro.csv, Storage.csv, Flex_demand.csv
        ```python
        # For other technologies, just use the technology name as Resource_Type
        df['Resource_Type'] = df['technology'].str.replace(' ', '_')
        ```
- added MGA
    - Added MGA = 1 in 
        - Thermal.csv 
        - Vre.csv
        - Must_run.csv
        - Hydro.csv
    - Added MGA = 0 in 
        - Storage.csv
        - Flex_demand.csv
## per scenario
- p4:
    - CO2Cap = 1 in genx_settings.yml
- p1 and p3:
    - CO2Cap = 0 in genx_settings.yml 

# task list
DONE: Try out MGA. You need to add MGA column to the vre and thermal file . 
DONE: isolate each policy category and see which one is causing the infeasibility problem. 
    - Infeasibility is likely due to the CO2 cap policy. Remove the CO2 cap policy for p1 and p2.
DONE: use gurobi to solve the model. some code needs to be uncommented . DONE.
    - DONE: BarObjRng is commented out in the gurobi_settings.yml file. 
    - DONE: run_genx_case!(dirname(@__FILE__),Gurobi.Optimizer) added to the run.jl file.
    - DONE: Changed the solver to Gurobi in the genx_settings.yml file.

DONE: remove the co2cap policy for the infeasible scenario p1 and p3
    storage and flex demand mga = 0
    add resource_type based on tech+cluster for vre 
    add resource type based on technology for others