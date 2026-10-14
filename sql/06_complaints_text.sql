DROP VIEW IF EXISTS complaints_text;

CREATE VIEW complaints_text AS
SELECT
    c.*,
    CASE
        -- Consumer Loan (2015-17) split by sub-product
        WHEN c.product = 'Consumer Loan' AND c.sub_product ILIKE 'vehicle%'
            THEN 'Vehicle loan'
        WHEN c.product = 'Consumer Loan'
            THEN 'Personal/payday loan'

        WHEN c.product IN (
            'Credit reporting',
            'Credit reporting, credit repair services, or other personal consumer reports',
            'Credit reporting or other personal consumer reports'
        ) THEN 'Credit reporting'

        WHEN c.product = 'Debt collection'
            THEN 'Debt collection'

        WHEN c.product = 'Mortgage'
            THEN 'Mortgage'

        WHEN c.product IN (
            'Bank account or service',
            'Checking or savings account'
        ) THEN 'Checking/savings'

        WHEN c.product IN (
            'Credit card',
            'Prepaid card',
            'Credit card or prepaid card'
        ) THEN 'Credit/prepaid card'

        WHEN c.product = 'Student loan'
            THEN 'Student loan'

        WHEN c.product = 'Vehicle loan or lease'
            THEN 'Vehicle loan'

        WHEN c.product IN (
            'Payday loan',
            'Payday loan, title loan, or personal loan',
            'Payday loan, title loan, personal loan, or advance loan'
        ) THEN 'Personal/payday loan'

        WHEN c.product IN (
            'Money transfers',
            'Virtual currency',
            'Money transfer, virtual currency, or money service'
        ) THEN 'Money transfer'

        WHEN c.product IN (
            'Other financial service',
            'Debt or credit management'
        ) THEN 'Other'
    END AS product_group,
    n.narrative
FROM complaints c
JOIN narratives n USING (complaint_id);