DROP TABLE IF EXISTS model_data;

CREATE TABLE model_data AS
SELECT
    complaint_id,
    date_received,
    product_group,
    narrative,
    CASE
        WHEN date_received <  '2024-01-01' THEN 'train'
        WHEN date_received <  '2025-01-01' THEN 'val'
        ELSE 'test'
    END AS split
FROM (
    SELECT DISTINCT ON (md5(narrative)) *
    FROM complaints_text
    ORDER BY md5(narrative), date_received      -- keeps the earliest copy
) dedup
WHERE array_length(regexp_split_to_array(narrative, '\s+'), 1) >= 10;

ALTER TABLE model_data ADD PRIMARY KEY (complaint_id);
CREATE INDEX ON model_data (split, product_group);
ANALYZE model_data;