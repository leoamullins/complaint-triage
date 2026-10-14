DROP TABLE IF EXISTS raw_complaints;

CREATE TABLE raw_complaints (
    date_received                 TEXT,
    product                       TEXT,
    sub_product                   TEXT,
    issue                         TEXT,
    sub_issue                     TEXT,
    company_public_response       TEXT,
    company                       TEXT,
    state                         TEXT,
    zip_code                      TEXT,
    tags                          TEXT,
    submitted_via                 TEXT,
    date_sent_to_company          TEXT,
    company_response_to_consumer  TEXT,
    timely_response               TEXT,
    complaint_id                  TEXT
);