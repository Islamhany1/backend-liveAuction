CREATE TABLE users (
                       id SERIAL PRIMARY KEY,
                       username VARCHAR(50) UNIQUE NOT NULL,
                       password VARCHAR(255) NOT NULL,
                       email VARCHAR(100) UNIQUE NOT NULL,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE items (
                       id SERIAL PRIMARY KEY,
                       title VARCHAR(100) NOT NULL,
                       description TEXT,
                       starting_price DECIMAL(10, 2) NOT NULL,
                       current_price DECIMAL(10, 2) NOT NULL,
                       status VARCHAR(20) DEFAULT 'ACTIVE',
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE bids (
                      id SERIAL PRIMARY KEY,
                      item_id INT REFERENCES items(id) ON DELETE CASCADE,
                      user_id INT REFERENCES users(id) ON DELETE CASCADE,
                      bid_amount DECIMAL(10, 2) NOT NULL,
                      bid_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);