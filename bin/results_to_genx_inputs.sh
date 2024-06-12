# bash

YEARS=( 2027 2030 2035 2040 2045 2050 ) # 2030 2040 2050  2027 2030 2035 2040 2045 2050
PERIODS=( "p1" "p2" "p3" "p4" "p5" "p6" ) # "p1" "p2" "p3"
folder=$1
model=$2

 if [[ "$folder" == *"base"* ]]; then
    input_folder="base_52_week_commit"
    # Thanks chatGPT!
    co2_slack=$(echo "$folder" | awk -F'-' '{for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+$/) {print $i; exit}}')
    echo "CO2 slack is ${co2_slack}"
else
    input_folder="current_policies_52_week_commit"
    co2_slack=200
fi

echo "${folder}"
echo "${input_folder}"
for i in "${!YEARS[@]}"; do

    RESULTS_PATH="../${folder}/${model}_results_summary"
    GENX_INPUTS_PATH="../genx-op-inputs/${input_folder}/Inputs/Inputs_${PERIODS[$i]}"
    OUTPUT_PATH="../${folder}/${model}_op_inputs/Inputs/Inputs_${PERIODS[$i]}"

    python results_to_genx_inputs.py "$RESULTS_PATH" "$GENX_INPUTS_PATH" "$OUTPUT_PATH" --co2-slack "${co2_slack}" --year "${YEARS[$i]}"
    
    # if test -f "${OUTPUT_PATH}/CO2_cap_slack.csv"; then 
    #     sed -i '' "s/200/${co2_slack}/g" "${OUTPUT_PATH}/CO2_cap_slack.csv"
    # fi

    # if test -f "${OUTPUT_PATH}/Generators_variability.csv"; then
    #     gzip -f "${OUTPUT_PATH}/Generators_variability.csv"
    # fi

    # if test -f "$GENX_INPUTS_PATH/Run.jl"; then
    #     cp -r "$GENX_INPUTS_PATH/Run.jl" "$OUTPUT_PATH"
    # else
    #     echo "No Run.jl file in ${GENX_INPUTS_PATH}"
    # fi
    # if test -d "$GENX_INPUTS_PATH/Settings"; then
    #     cp -r "$GENX_INPUTS_PATH/Settings" "$OUTPUT_PATH"
    # else
    #     echo "No Settings folder in ${GENX_INPUTS_PATH}"
    # fi
done