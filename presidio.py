from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
faker = Faker()

def anonymize_text(text, technique):
    analysis_results = analyzer.analyze(text=text, entities=None, language='en')
    operators = {}
    counter = {}

    if technique == "replace":
        for result in analysis_results:
            entity = result.entity_type
            count = counter.get(entity, 0) + 1
            counter[entity] = count
            if entity == "PERSON":
                fake_value = faker.name()
            elif entity in ["LOCATION", "GPE"]:
                fake_value = faker.city()
            elif entity == "EMAIL_ADDRESS":
                fake_value = faker.email()
            elif entity == "PHONE_NUMBER":
                fake_value = faker.phone_number()
            elif entity == "CREDIT_CARD":
                fake_value = faker.credit_card_number()
            elif entity == "IBAN_CODE":
                fake_value = faker.iban()
            elif entity == "IP_ADDRESS":
                fake_value = faker.ipv4()
            elif entity == "DATE_TIME":
                fake_value = faker.date()
            elif entity == "URL":
                fake_value = faker.url()
            elif entity == "US_SSN":
                fake_value = faker.ssn()
            elif entity == "MEDICAL_LICENSE":
                fake_value = faker.bothify(text="ML-######")
            elif entity == "US_DRIVER_LICENSE":
                fake_value = faker.bothify(text="D###-####-####")
            elif entity == "US_PASSPORT":
                fake_value = faker.bothify(text="#########")
            elif entity == "CRYPTO":
                fake_value = faker.sha256()[:34]
            elif entity == "NRP":
                fake_value = faker.country()
            else:
                fake_value = f"{entity}_{count}"

            operators[entity] = OperatorConfig("replace", {"new_value": fake_value})

    elif technique == "mask":
        for result in analysis_results:
            entity = result.entity_type
            operators[entity] = OperatorConfig("replace", {"new_value": entity})

    elif technique == "redact":
        for result in analysis_results:
            entity = result.entity_type
            operators[entity] = OperatorConfig("replace", {"new_value": "REDACTED"})

    else:
        print("Invalid technique selected. Defaulting to redaction.")
        for result in analysis_results:
            entity = result.entity_type
            operators[entity] = OperatorConfig("replace", {"new_value": "REDACTED"})

    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=analysis_results,
        operators=operators
    )

    return anonymized_result.text
