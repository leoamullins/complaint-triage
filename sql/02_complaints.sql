DROP TABLE IF EXISTS complaints;

CREATE TABLE complaints AS
SELECT
    NULLIF(complaint_id, '')::BIGINT               AS complaint_id,
    NULLIF(date_received, '')::DATE                AS date_received,
    NULLIF(date_sent_to_company, '')::DATE         AS date_sent_to_company,
    NULLIF(product, '')                            AS product,
    NULLIF(sub_product, '')                        AS sub_product,
    NULLIF(issue, '')                              AS issue,
    NULLIF(sub_issue, '')                          AS sub_issue,
    NULLIF(company, '')                            AS company,
    NULLIF(company_public_response, '')            AS company_public_response,
    NULLIF(company_response_to_consumer, '')       AS company_response_to_consumer,
    CASE timely_response
        WHEN 'Yes' THEN TRUE
        WHEN 'No'  THEN FALSE
    END                                            AS timely_response,
    NULLIF(state, '')                              AS state,
    NULLIF(zip_code, '')                           AS zip_code,
    NULLIF(tags, '')                               AS tags,
    NULLIF(submitted_via, '')                      AS submitted_via
FROM raw_complaints;

ALTER TABLE complaints ADD PRIMARY KEY (complaint_id);

CREATE INDEX idx_complaints_date    ON complaints (date_received);
CREATE INDEX idx_complaints_product ON complaints (product);
CREATE INDEX idx_complaints_company ON complaints (company);

ANALYZE complaints;