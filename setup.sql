create or replace database airbnb;

create or replace table airbnb.public.airbnb as
(
  select * from
  (
    select a.* exclude CATEGORY_RATING, 
    f.value:value::text as rating, 
    f.value:name::text as category_name from
    (
      select * exclude (price,LONG),
      LONG as lon,
      SNOWFLAKE.CORTEX.EXTRACT_ANSWER(price,'What is the value?')[0]:answer::text as price,
      SNOWFLAKE.CORTEX.EXTRACT_ANSWER(house_rules,'What is the check in time?')[0]:answer::text as check_in, 
      SNOWFLAKE.CORTEX.EXTRACT_ANSWER(house_rules,'What is the check out time?')[0]:answer::text check_out,
      try_to_number(SNOWFLAKE.CORTEX.EXTRACT_ANSWER(details,'What is the number of bedrooms?')[0]:answer::text) as bedrooms, 
      try_to_number(SNOWFLAKE.CORTEX.EXTRACT_ANSWER(details,'What is the number of baths?')[0]:answer::text) as baths,
      SNOWFLAKE.CORTEX.SUMMARIZE(reviews) as summary,
      SNOWFLAKE.CORTEX.EMBED_TEXT('e5-base-v2', reviews) as embedding,
      SNOWFLAKE.CORTEX.SENTIMENT(reviews) as sentiment 
      from airbnb_properties_information.public.airbnb_properties_information
    ) a, lateral flatten(parse_json(a.category_rating)) f
   )
   pivot(min(RATING) for category_name in ('Cleanliness','Accuracy', 'Communication', 'Location', 'Check-in', 'Value') )
   AS p (TIMESTAMP, URL, AMENITIES, AVAILABILITY, AVAILABLE_DATES, CATEGORY, DESCRIPTION, DESCRIPTION_ITEMS, DETAILS, FINAL_URL, 
   GUESTS, HIGHLIGHTS, HOUSE_RULES, IMAGE, IMAGES, LAT, NAME, PETS_ALLOWED, 
   RATINGS, REVIEWS, SELLER_INFO, LON, PRICE, CHECK_IN, CHECK_OUT, BEDROOMS,BATHS, 
   SUMMARY, EMBEDDING, SENTIMENT, Cleanliness_r,Accuracy_r, Communication_r, Location_r, Checkin_r, Value_r)
);

alter table airbnb.public.airbnb cluster by (guests, price, availability);
