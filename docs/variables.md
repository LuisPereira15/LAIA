# Dataset Variables — BTS Airline On-Time Performance Data

| Variable | Type | Description |
|---|---|---|
| YEAR | int | Year of the flight record (2022 or 2023) |
| QUARTER | int | Quarter of the year (1–4) |
| MONTH | int | Month of the year (1–12) |
| DAY_OF_WEEK | int | Day of the week (1=Monday through 7=Sunday) |
| OP_CARRIER_AIRLINE_ID | int | Unique identifier for the operating airline |
| OP_CARRIER_FL_NUM | int | Flight number assigned by the operating carrier |
| ORIGIN | str | Origin airport code (e.g., "ATL", "LAX") |
| ORIGIN_STATE_ABR | str | US state abbreviation of the origin airport (e.g., "GA", "CA") |
| DEST | str | Destination airport code |
| DEST_STATE_ABR | str | US state abbreviation of the destination airport |
| CRS_DEP_TIME | int | Scheduled departure time (local, HHMM format, e.g., 1430 = 2:30 PM) |
| DEP_DELAY | float | Departure delay in minutes (negative = early departure). Missing for cancelled flights. |
| CRS_ARR_TIME | int | Scheduled arrival time (local, HHMM format) |
| ARR_DELAY_NEW | float | Arrival delay in minutes (0 if early/on-time, positive if late). Missing for cancelled/diverted flights. **Target variable is derived from this.** |
| CANCELLED | float | Whether the flight was cancelled (1.0 = cancelled, 0.0 = not cancelled) |
| DIVERTED | float | Whether the flight was diverted (1.0 = diverted, 0.0 = not diverted) |
| CRS_ELAPSED_TIME | float | Scheduled flight duration in minutes (gate to gate) |
| DISTANCE | float | Distance between origin and destination in miles |
| DISTANCE_GROUP | int | Distance group (1 = less than 250 miles, each group adds 250 miles) |

## Target Variable (derived)

Derived from `ARR_DELAY_NEW`:

| Category | Condition |
|---|---|
| `on_time` | ARR_DELAY_NEW < 15 minutes |
| `moderate_delay` | 15 <= ARR_DELAY_NEW < 60 minutes |
| `severe_delay` | ARR_DELAY_NEW >= 60 minutes |

## Features NOT usable for prediction

Only information available **before scheduled departure** may be used. These columns are known only after the flight operates and must be excluded from model input:

- `DEP_DELAY` — actual departure delay
- `ARR_DELAY_NEW` — actual arrival delay (used only to derive the target)
- `CANCELLED` — whether the flight was cancelled (only used for filtering)
- `DIVERTED` — whether the flight was diverted (only used for filtering)