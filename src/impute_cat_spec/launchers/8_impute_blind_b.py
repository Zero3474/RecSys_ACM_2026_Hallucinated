import polars as pl 
import json

from resources import (
    BLIND_A,
    BLIND_B,
    ENSEMBLE_DIR,
    OUPUT_DIR
)

def validate_blind_datasets(ba: pl.DataFrame, bb: pl.DataFrame, common: pl.DataFrame) -> pl.DataFrame:
    print("--- Starting Data Quality Checks ---")
    
    # 1. Check for null or empty user_ids in Blind A and Blind B
    empty_condition = pl.col("user_id").is_null() | (pl.col("user_id") == "")
    
    empty_a_count = ba.filter(empty_condition).height
    empty_b_count = bb.filter(empty_condition).height
    
    print(f"1. Empty/Null user_id in Blind A: {empty_a_count}")
    print(f"1. Empty/Null user_id in Blind B: {empty_b_count}")
    
    def user_profile_match(row):        
        if row.get("user_id") is None or row.get("user_id") == "":
            return True  # If user_id is empty/null, we don't care about profile mismatch
        
        if row.get("user_id") != row.get("user_id_a"):
            raise ValueError(f"user_id mismatch: Blind A has {row.get('user_id_a')}, Blind B has {row.get('user_id')}")

        user_a = row.get("user_profile_a")
        user_b = row.get("user_profile")

        # Compare each field in the user profiles
        for key in set(user_a.keys()+user_b.keys()):
            if user_a.get(key) != user_b.get(key):
                raise ValueError(f"user_profile mismatch for key '{key}': Blind A has {user_a.get(key)}, Blind B has {user_b.get(key)}")

        return True

    def conv_match(row):
        conv_a = row.get("conversations_a")
        conv_b = row.get("conversations")
        
        # If either is completely missing or not a list, we can't compare
        if conv_a is None or conv_b is None or not isinstance(conv_a, list) or not isinstance(conv_b, list):
            return False
            
        min_len = min(len(conv_a), len(conv_b))
        
        # If both are empty, they technically match
        if min_len == 0:
            return True
            
        # Iterate through the shared length, just like your snippet
        for i in range(min_len):
            turn_a = conv_a[i]
            turn_b = conv_b[i]
            
            # Ensure the turns are dictionaries (structs)
            if not isinstance(turn_a, dict) or not isinstance(turn_b, dict):
                continue
                
            role_a = turn_a.get("role")
            role_b = turn_b.get("role")
            
            if role_a != role_b:
                return False

            # If BOTH are user turns, their content MUST match
            elif role_a == "user":
                cont_a = turn_a.get("content")
                cont_b = turn_b.get("content")
                
                if cont_a != cont_b:
                    return False # Immediate failure if content mismatches

            # PANIC if music content mismatches, raise an error
            elif role_a == "music":
                cont_a = turn_a.get("content")
                cont_b = turn_b.get("content")
                
                if cont_a != cont_b:
                    raise ValueError(f"Music content mismatch at index {i}: Blind A has '{cont_a}', Blind B has '{cont_b}'")

        return True

    # Apply the checks to the common dataframe
    common_checked = common.with_columns(
        is_same_user=pl.struct(["user_profile", "user_profile_a"]).map_elements(
            user_profile_match, 
            return_dtype=pl.Boolean
        ),
        is_same_conv=pl.struct(["conversations", "conversations_a"]).map_elements(
            conv_match, 
            return_dtype=pl.Boolean
        )
    )
    
    # Summarize Results
    mismatched_users = common_checked.filter(~pl.col("is_same_user")).height
    mismatched_convs = common_checked.filter(~pl.col("is_same_conv")).height
    
    print(f"2. Common dataset length: {len(common)}")
    print(f"2. Mismatched/Invalid user_ids in common dataset: {mismatched_users}")
    print(f"3. Mismatched conversations in common dataset: {mismatched_convs}")
    print("------------------------------------")
    
    return common_checked

def main():
    # Load Blind A and Blind B datasets 
    ba = pl.read_parquet(BLIND_A)
    bb = pl.read_parquet(BLIND_B)
    
    blind_b_predictions_file = ENSEMBLE_DIR / "final_blind_b_predictions.json"
    if not blind_b_predictions_file.exists():
        raise FileNotFoundError(f"Blind B predictions file not found: {blind_b_predictions_file} \nPlease ensure that the ensemble predictions have been generated before running this script.")
    
    with open(blind_b_predictions_file, "r") as f:
        data = json.load(f)
    
        imputations = pl.DataFrame([
            {"session_id": sid, **preds} for sid, preds in data.items()
        ])

    print("Blind A: ", ba.columns, " | Length: ", len(ba))
    print("Blind B: ", bb.columns, " | Length: ", len(bb))
    print("Imputations: ", imputations.columns, " | Length: ", len(imputations))

    # Identify the common session_ids between Blind A and Blind B
    common = bb.select("session_id", "user_id", "user_profile", "conversations").join(ba, on="session_id", how="inner", suffix="_a")

    # Recover what we can from Blind A and validate the common sessions
    common_checked = validate_blind_datasets(ba, bb, common).select([
        "session_id",
        "user_id_a",           # Blind A
        "session_date",        # Blind A
        "user_profile_a",      # Blind A
        "conversation_goal",   # Blind A
        "goal_progress_assessments", # Blind A
        "conversations",       # Blind B
        
        # Additional columns for validation
        "conversations_a",     # Blind A
        "is_same_user",
        "is_same_conv",
    ]).rename({
        "user_id_a": "user_id",
        "user_profile_a": "user_profile",
        "conversations_a": "conversations_blind_a"
    }).with_columns(
        from_blind_a=pl.lit(True),
        imputed=pl.lit(False)
    )

    print("Common Checked: ", common_checked.columns, " | Length: ", len(common_checked))

    # Merge the imputations with Blind B, ensuring we only impute for sessions that were not recoverable from Blind A
    imputed_df = bb.join(
        # Select only the session_id that are NOT in the common_checked (i.e., those that need imputation)
        common_checked.select(["session_id"]), on="session_id", how="anti"
    ).join(imputations, on="session_id", how="left")

    # Package the ML Imputations into the exact same struct format
    imputed_df = imputed_df.with_columns(
        conversation_goal=pl.struct([
            pl.col("category").cast(pl.String),
            pl.lit("").alias("listener_goal"),
            pl.col("specificity").cast(pl.String)
        ])
    ).drop(
        # Drop the raw imputation columns so they don't clutter the final output
        ["category", "specificity"] 
    ).with_columns(
        imputed=pl.lit(True),
        from_blind_a=pl.lit(False)
    )

    # Merge the two datasets: those recoverable from Blind A and those imputed from Blind B
    final_df = pl.concat([common_checked, imputed_df], how="diagonal")

    print("Final Output: ", final_df.columns, " | Length: ", len(final_df))
    
    OUPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(OUPUT_DIR / "final_blind_b.parquet")


if __name__ == "__main__":
    main()
