DROP TABLE IF EXISTS narratives;

CREATE TABLE narratives AS
SELECT DISTINCT ON (complaint_id::BIGINT)
    complaint_id::BIGINT          AS complaint_id,
    consumer_complaint_narrative  AS narrative
FROM raw_narratives
WHERE consumer_complaint_narrative <> ''
ORDER BY complaint_id::BIGINT;

ALTER TABLE narratives ADD PRIMARY KEY (complaint_id);
ANALYZE narratives;