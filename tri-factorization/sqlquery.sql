
CREATE VIEW usersMostRatings AS
SELECT user_id, COUNT(1) cantidad FROM mostRatingsByMovie
GROUP BY user_id
ORDER BY cantidad DESC;

SELECT t1.* FROM ratings t1
INNER JOIN (SELECT user_id FROM usersMostRatings
WHERE cantidad > 40
ORDER BY rand()
LIMIT 500) t2 ON t1.user_id = t2.user_id