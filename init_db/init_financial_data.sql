-- Create financial_data table
CREATE TABLE IF NOT EXISTS financial_data (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(100),
    ticker VARCHAR(10),
    sector VARCHAR(50),
    market_cap DECIMAL(15,2),
    revenue DECIMAL(15,2),
    pe_ratio DECIMAL(10,2),
    dividend_yield DECIMAL(5,2)
);

-- Insert sample financial data
INSERT INTO financial_data (company_name, ticker, sector, market_cap, revenue, pe_ratio, dividend_yield) VALUES
('Apple Inc', 'AAPL', 'Technology', 2500000000000.00, 394328000000.00, 28.50, 0.50),
('Microsoft Corp', 'MSFT', 'Technology', 2800000000000.00, 198270000000.00, 32.10, 0.80),
('Amazon.com Inc', 'AMZN', 'Consumer', 1800000000000.00, 514004000000.00, 45.20, 0.00),
('Alphabet Inc', 'GOOGL', 'Technology', 1700000000000.00, 307394000000.00, 25.80, 0.00),
('Tesla Inc', 'TSLA', 'Consumer', 800000000000.00, 81462000000.00, 65.40, 0.00),
('Meta Platforms', 'META', 'Technology', 900000000000.00, 116609000000.00, 18.90, 0.00),
('Netflix Inc', 'NFLX', 'Consumer', 250000000000.00, 31616000000.00, 35.60, 0.00),
('Adobe Inc', 'ADBE', 'Technology', 200000000000.00, 19409000000.00, 42.30, 0.00),
('Salesforce Inc', 'CRM', 'Technology', 220000000000.00, 31135000000.00, 28.70, 0.00),
('Oracle Corp', 'ORCL', 'Technology', 300000000000.00, 49954000000.00, 22.40, 1.20),
('NVIDIA Corp', 'NVDA', 'Technology', 1200000000000.00, 26974000000.00, 85.20, 0.20),
('Intel Corp', 'INTC', 'Technology', 200000000000.00, 63054000000.00, 15.80, 2.50),
('Cisco Systems', 'CSCO', 'Technology', 180000000000.00, 51557000000.00, 12.30, 3.10),
('IBM Corp', 'IBM', 'Technology', 120000000000.00, 60530000000.00, 18.90, 4.20),
('HP Inc', 'HPQ', 'Technology', 30000000000.00, 63200000000.00, 8.50, 3.80),
('Dell Technologies', 'DELL', 'Technology', 50000000000.00, 101200000000.00, 12.10, 2.10),
('VMware Inc', 'VMW', 'Technology', 60000000000.00, 12850000000.00, 25.40, 0.00),
('ServiceNow Inc', 'NOW', 'Technology', 100000000000.00, 8500000000.00, 45.60, 0.00),
('Workday Inc', 'WDAY', 'Technology', 70000000000.00, 5800000000.00, 38.90, 0.00),
('Snowflake Inc', 'SNOW', 'Technology', 80000000000.00, 2500000000.00, 120.50, 0.00); 