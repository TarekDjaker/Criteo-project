-- SQL Analytics Templates for Criteo Interview
-- Focus: Window Functions, CTEs, Performance Optimization

-- ========================================
-- 1. RUNNING SUM PATTERN (CTR accumulation)
-- ========================================
WITH daily_metrics AS (
    SELECT
        date,
        campaign_id,
        SUM(clicks) as daily_clicks,
        SUM(impressions) as daily_impressions,
        SUM(clicks) * 1.0 / NULLIF(SUM(impressions), 0) as daily_ctr
    FROM ad_events
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY date, campaign_id
)
SELECT
    date,
    campaign_id,
    daily_clicks,
    daily_impressions,
    daily_ctr,
    -- Running totals
    SUM(daily_clicks) OVER (
        PARTITION BY campaign_id
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_clicks,
    -- Moving average (7-day)
    AVG(daily_ctr) OVER (
        PARTITION BY campaign_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as ctr_7d_avg,
    -- Rank by performance
    DENSE_RANK() OVER (
        PARTITION BY date
        ORDER BY daily_ctr DESC
    ) as daily_performance_rank
FROM daily_metrics
ORDER BY campaign_id, date;

-- ========================================
-- 2. ROW_NUMBER + ANTI-JOIN (Deduplication)
-- ========================================
WITH ranked_clicks AS (
    SELECT
        user_id,
        ad_id,
        timestamp,
        click_value,
        -- Deduplicate by most recent click
        ROW_NUMBER() OVER (
            PARTITION BY user_id, ad_id
            ORDER BY timestamp DESC
        ) as rn
    FROM user_clicks
    WHERE timestamp >= CURRENT_DATE - INTERVAL '1 day'
)
-- Get unique clicks
SELECT
    user_id,
    ad_id,
    timestamp,
    click_value
FROM ranked_clicks
WHERE rn = 1

-- Anti-join pattern for finding unmatched records
UNION ALL

SELECT
    u.user_id,
    NULL as ad_id,
    u.last_active as timestamp,
    0 as click_value
FROM users u
LEFT JOIN ranked_clicks rc ON u.user_id = rc.user_id
WHERE rc.user_id IS NULL
  AND u.last_active >= CURRENT_DATE - INTERVAL '7 days';

-- ========================================
-- 3. ROLLING 7-DAY WINDOW (Revenue analysis)
-- ========================================
WITH daily_revenue AS (
    SELECT
        date,
        advertiser_id,
        SUM(revenue) as daily_revenue,
        COUNT(DISTINCT campaign_id) as active_campaigns
    FROM ad_revenue
    GROUP BY date, advertiser_id
)
SELECT
    date,
    advertiser_id,
    daily_revenue,
    -- 7-day rolling sum
    SUM(daily_revenue) OVER (
        PARTITION BY advertiser_id
        ORDER BY date
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    ) as revenue_7d,
    -- 7-day rolling average
    AVG(daily_revenue) OVER (
        PARTITION BY advertiser_id
        ORDER BY date
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    ) as avg_revenue_7d,
    -- Week-over-week comparison
    daily_revenue - LAG(daily_revenue, 7) OVER (
        PARTITION BY advertiser_id
        ORDER BY date
    ) as wow_change,
    -- Percent change
    CASE
        WHEN LAG(daily_revenue, 7) OVER (PARTITION BY advertiser_id ORDER BY date) > 0
        THEN (daily_revenue - LAG(daily_revenue, 7) OVER (PARTITION BY advertiser_id ORDER BY date)) * 100.0
             / LAG(daily_revenue, 7) OVER (PARTITION BY advertiser_id ORDER BY date)
        ELSE NULL
    END as wow_percent_change
FROM daily_revenue
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY advertiser_id, date;

-- ========================================
-- 4. COHORT ANALYSIS (User retention)
-- ========================================
WITH user_cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('week', first_click_date) as cohort_week,
        first_click_date
    FROM (
        SELECT
            user_id,
            MIN(click_date) as first_click_date
        FROM user_clicks
        GROUP BY user_id
    ) first_clicks
),
cohort_activity AS (
    SELECT
        c.cohort_week,
        DATEDIFF('week', c.cohort_week, DATE_TRUNC('week', a.click_date)) as weeks_since_start,
        COUNT(DISTINCT c.user_id) as active_users
    FROM user_cohorts c
    JOIN user_clicks a ON c.user_id = a.user_id
    WHERE a.click_date >= c.first_click_date
    GROUP BY c.cohort_week, weeks_since_start
),
cohort_sizes AS (
    SELECT
        cohort_week,
        COUNT(DISTINCT user_id) as cohort_size
    FROM user_cohorts
    GROUP BY cohort_week
)
SELECT
    ca.cohort_week,
    ca.weeks_since_start,
    ca.active_users,
    cs.cohort_size,
    ca.active_users * 100.0 / cs.cohort_size as retention_rate
FROM cohort_activity ca
JOIN cohort_sizes cs ON ca.cohort_week = cs.cohort_week
WHERE ca.weeks_since_start <= 12  -- 12 week retention
ORDER BY ca.cohort_week, ca.weeks_since_start;

-- ========================================
-- 5. PERCENTILE ANALYSIS (Bid distribution)
-- ========================================
WITH bid_stats AS (
    SELECT
        advertiser_id,
        campaign_id,
        bid_amount,
        -- Percentile calculation
        PERCENT_RANK() OVER (
            PARTITION BY campaign_id
            ORDER BY bid_amount
        ) as bid_percentile,
        -- Quartiles
        NTILE(4) OVER (
            PARTITION BY campaign_id
            ORDER BY bid_amount
        ) as bid_quartile
    FROM bids
    WHERE bid_date >= CURRENT_DATE - INTERVAL '7 days'
)
SELECT
    campaign_id,
    COUNT(*) as total_bids,
    MIN(bid_amount) as min_bid,
    MAX(bid_amount) as max_bid,
    AVG(bid_amount) as avg_bid,
    -- Median (P50)
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bid_amount) as median_bid,
    -- P25, P75, P95
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bid_amount) as p25_bid,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bid_amount) as p75_bid,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY bid_amount) as p95_bid,
    -- IQR for outlier detection
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bid_amount) -
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bid_amount) as iqr
FROM bid_stats
GROUP BY campaign_id;

-- ========================================
-- 6. FUNNEL ANALYSIS (Conversion path)
-- ========================================
WITH user_journey AS (
    SELECT
        user_id,
        event_type,
        event_timestamp,
        -- Assign step number
        CASE event_type
            WHEN 'impression' THEN 1
            WHEN 'click' THEN 2
            WHEN 'add_to_cart' THEN 3
            WHEN 'purchase' THEN 4
        END as funnel_step,
        -- Session identification
        SUM(CASE
            WHEN event_timestamp - LAG(event_timestamp) OVER (PARTITION BY user_id ORDER BY event_timestamp) > INTERVAL '30 minutes'
            THEN 1
            ELSE 0
        END) OVER (PARTITION BY user_id ORDER BY event_timestamp) as session_id
    FROM user_events
    WHERE event_date >= CURRENT_DATE - INTERVAL '7 days'
),
funnel_metrics AS (
    SELECT
        funnel_step,
        COUNT(DISTINCT user_id) as users_reached,
        COUNT(DISTINCT CONCAT(user_id, '_', session_id)) as sessions_reached
    FROM user_journey
    WHERE funnel_step IS NOT NULL
    GROUP BY funnel_step
)
SELECT
    funnel_step,
    CASE funnel_step
        WHEN 1 THEN 'Impression'
        WHEN 2 THEN 'Click'
        WHEN 3 THEN 'Add to Cart'
        WHEN 4 THEN 'Purchase'
    END as step_name,
    users_reached,
    sessions_reached,
    -- Conversion from previous step
    users_reached * 100.0 / LAG(users_reached) OVER (ORDER BY funnel_step) as step_conversion_rate,
    -- Overall conversion from start
    users_reached * 100.0 / FIRST_VALUE(users_reached) OVER (ORDER BY funnel_step) as overall_conversion_rate
FROM funnel_metrics
ORDER BY funnel_step;

-- ========================================
-- 7. ATTRIBUTION MODELING (Last-click vs Multi-touch)
-- ========================================
WITH touchpoints AS (
    SELECT
        conversion_id,
        user_id,
        channel,
        touchpoint_timestamp,
        conversion_value,
        -- Order touchpoints
        ROW_NUMBER() OVER (PARTITION BY conversion_id ORDER BY touchpoint_timestamp) as touch_order,
        ROW_NUMBER() OVER (PARTITION BY conversion_id ORDER BY touchpoint_timestamp DESC) as reverse_order,
        COUNT(*) OVER (PARTITION BY conversion_id) as total_touches
    FROM conversion_touchpoints
    WHERE conversion_date >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT
    channel,
    -- Last-click attribution
    SUM(CASE WHEN reverse_order = 1 THEN conversion_value ELSE 0 END) as last_click_value,
    -- First-click attribution
    SUM(CASE WHEN touch_order = 1 THEN conversion_value ELSE 0 END) as first_click_value,
    -- Linear attribution (equal credit)
    SUM(conversion_value * 1.0 / total_touches) as linear_value,
    -- Time-decay attribution (40% last, 30% second-last, 20% third-last, 10% rest)
    SUM(CASE
        WHEN reverse_order = 1 THEN conversion_value * 0.4
        WHEN reverse_order = 2 THEN conversion_value * 0.3
        WHEN reverse_order = 3 THEN conversion_value * 0.2
        ELSE conversion_value * 0.1 / NULLIF(total_touches - 3, 0)
    END) as time_decay_value,
    -- U-shaped attribution (40% first, 40% last, 20% middle)
    SUM(CASE
        WHEN touch_order = 1 THEN conversion_value * 0.4
        WHEN reverse_order = 1 THEN conversion_value * 0.4
        ELSE conversion_value * 0.2 / NULLIF(total_touches - 2, 0)
    END) as u_shaped_value
FROM touchpoints
GROUP BY channel
ORDER BY last_click_value DESC;

-- ========================================
-- 8. SEGMENT PERFORMANCE (A/B Test Analysis)
-- ========================================
WITH experiment_metrics AS (
    SELECT
        experiment_id,
        variant,
        user_id,
        SUM(clicks) as user_clicks,
        SUM(conversions) as user_conversions,
        SUM(revenue) as user_revenue
    FROM experiment_results
    WHERE experiment_date >= CURRENT_DATE - INTERVAL '14 days'
    GROUP BY experiment_id, variant, user_id
),
variant_stats AS (
    SELECT
        experiment_id,
        variant,
        COUNT(DISTINCT user_id) as users,
        SUM(user_clicks) as total_clicks,
        SUM(user_conversions) as total_conversions,
        SUM(user_revenue) as total_revenue,
        AVG(user_clicks) as avg_clicks_per_user,
        AVG(user_conversions) as avg_conversions_per_user,
        AVG(user_revenue) as avg_revenue_per_user,
        -- Statistical measures
        STDDEV(user_revenue) as revenue_stddev,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY user_revenue) as median_revenue
    FROM experiment_metrics
    GROUP BY experiment_id, variant
)
SELECT
    vs.*,
    -- Lift calculations (assuming 'control' is baseline)
    CASE
        WHEN variant != 'control' THEN
            (avg_revenue_per_user -
             FIRST_VALUE(avg_revenue_per_user) OVER (PARTITION BY experiment_id ORDER BY variant)) * 100.0 /
             NULLIF(FIRST_VALUE(avg_revenue_per_user) OVER (PARTITION BY experiment_id ORDER BY variant), 0)
        ELSE 0
    END as revenue_lift_percent,
    -- Statistical significance (simplified Z-score)
    CASE
        WHEN variant != 'control' AND revenue_stddev > 0 THEN
            ABS(avg_revenue_per_user -
                FIRST_VALUE(avg_revenue_per_user) OVER (PARTITION BY experiment_id ORDER BY variant)) /
            SQRT(revenue_stddev * revenue_stddev / users +
                 FIRST_VALUE(revenue_stddev * revenue_stddev / users) OVER (PARTITION BY experiment_id ORDER BY variant))
        ELSE NULL
    END as z_score
FROM variant_stats vs
ORDER BY experiment_id, variant;

-- ========================================
-- 9. OPTIMIZATION: Query Performance Tips
-- ========================================
/*
Performance Best Practices for Criteo Scale:

1. INDEXING STRATEGY:
   - Create indexes on: (date, campaign_id), (user_id, timestamp)
   - Composite indexes for JOIN conditions
   - Covering indexes for frequent SELECT columns

2. PARTITIONING:
   - Partition by date for time-series data
   - Hash partition by user_id for user-centric queries
   - Range partition for numerical IDs

3. QUERY OPTIMIZATION:
   - Use CTEs instead of subqueries when reusing results
   - Push filters down to earliest possible point
   - Avoid SELECT * - specify needed columns
   - Use EXPLAIN ANALYZE to check execution plans

4. WINDOW FUNCTION OPTIMIZATION:
   - Minimize PARTITION BY columns
   - Use ROWS instead of RANGE when possible
   - Consider materialized views for complex windows

5. JOIN OPTIMIZATION:
   - Join on indexed columns
   - Start with smallest table
   - Use EXISTS instead of IN for large lists
   - Consider denormalization for frequent joins

6. AGGREGATION OPTIMIZATION:
   - Pre-aggregate in CTEs
   - Use approximate functions (HyperLogLog for COUNT DISTINCT)
   - Consider sampling for large datasets

Example optimized query:
*/

-- Before optimization
SELECT user_id, COUNT(DISTINCT ad_id)
FROM (
    SELECT * FROM clicks
    WHERE date >= '2024-01-01'
) c
GROUP BY user_id;

-- After optimization
SELECT /*+ PARALLEL(4) */
    user_id,
    APPROX_COUNT_DISTINCT(ad_id) as unique_ads -- HyperLogLog
FROM clicks
WHERE date >= '2024-01-01'
    AND date < '2024-02-01'  -- Use upper bound
    AND user_id IS NOT NULL   -- Filter NULLs early
GROUP BY user_id;

-- ========================================
-- 10. REAL-TIME METRICS (Streaming SQL)
-- ========================================
/*
For real-time processing at Criteo scale:

CREATE MATERIALIZED VIEW real_time_ctr AS
SELECT
    DATE_TRUNC('minute', event_time) as minute,
    campaign_id,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) as clicks,
    SUM(CASE WHEN event_type = 'impression' THEN 1 ELSE 0 END) as impressions,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(CASE WHEN event_type = 'impression' THEN 1 ELSE 0 END), 0) as ctr
FROM kafka_stream_events
WHERE event_time >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY DATE_TRUNC('minute', event_time), campaign_id
WITH (
    timecol = 'minute',
    refresh_interval = '10 seconds'
);
*/