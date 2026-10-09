from datetime import datetime


def convert_date(value, is_birth=False):
    """
    Convert YYMMDD into YYYY-MM-DD.

    ICAO MRZ uses two-digit years:
      - expiry dates are always 20YY
      - birth dates are 20YY unless that would be in the future (then 19YY)
    """

    year = int(value[0:2])
    month = int(value[2:4])
    day = int(value[4:6])

    full_year = 2000 + year

    if is_birth and datetime(full_year, month, day) > datetime.now():
        full_year -= 100

    return f"{full_year:04d}-{month:02d}-{day:02d}"


def parse_mrz(line1, line2):

    # Issuing country
    issuing_country = line1[2:5]

    # Name field: SURNAME<<GIVEN<NAMES
    name_field = line1[5:44]
    parts = name_field.split("<<", 1)

    surname = parts[0].replace("<", " ").strip()

    if len(parts) > 1:
        given_names = parts[1].replace("<", " ").strip()
    else:
        given_names = ""

    # Passport number
    passport_number = line2[0:9].replace("<", "")

    # Nationality
    nationality = line2[10:13]

    # Date of birth
    date_of_birth = convert_date(line2[13:19], is_birth=True)

    # Sex (M / F / X, "<" = unspecified)
    sex = line2[20]
    if sex == "<":
        sex = ""

    # Expiry date
    expiry_date = convert_date(line2[21:27])

    return {
        "surname": surname,
        "given_names": given_names,
        "passport_number": passport_number,
        "nationality": nationality,
        "date_of_birth": date_of_birth,
        "sex": sex,
        "expiry_date": expiry_date,
        "issuing_country": issuing_country,
    }