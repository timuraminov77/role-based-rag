CREATE DATABASE IF NOT EXISTS rag;
USE rag;

CREATE TABLE IF NOT EXISTS users (
  user_id INT NOT NULL AUTO_INCREMENT,
  login VARCHAR(15) NOT NULL,
  password VARCHAR(15) NOT NULL,
  role VARCHAR(45) NOT NULL,
  PRIMARY KEY (user_id),
  UNIQUE KEY login_UNIQUE (login)
);

INSERT INTO users (user_id, login, password, role) VALUES
(1, 'sergey', 'sergey', 'finance'),
(2, 'artem',  'artem',  'general'),
(3, 'anna',   'anna',   'hr'),
(4, 'user',   'user',   'marketing'),
(5, 'timur',  'timur',  'engineering')
ON DUPLICATE KEY UPDATE
  password=VALUES(password),
  role=VALUES(role);
