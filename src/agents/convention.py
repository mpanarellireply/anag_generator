CONVENTION = """ Always follow these coding conventions:
- Do not produce any procedure
- Whenever you see the following select statement:
    SELECT CERR,
           XERR,
           FLG_ATTIVO
    INTO ...
  
  Do not use any aggregator function (MAX, MIN, COUNT, etc.) on the selected columns, leave them as they are.
- Naming convention
    1. All initialized variable inside the sql file must follow the pattern V_{{variable_name}}. 
    2. Input function parameters follow the pattern V_{{variable_name}} as well. 
- Exception convention: 
    1. The final excpetion block at the end of the function must always be: 
        WHEN OTHERS THEN 
            RETURN NVL(V_CERR,'') || ' - ERRORE NON GESTITO - ';
    2. Do not add any exception handling on the select statements that starts as follows:
       SELECT CERR,
              XERR,
              FLG_ATTIVO
       INTO ...
- Do not use any DEFAULT for input function parameters
"""
