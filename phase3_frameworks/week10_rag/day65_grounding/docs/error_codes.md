# Error Code Reference

## E401 Authentication Failed

Your API key is invalid or expired. Generate a new key in the dashboard
under Settings > API Access. Keys expire after 90 days.

## E429 Rate Limit Exceeded

You have sent too many requests. The Basic tier allows 100 requests per
minute. Wait 60 seconds before retrying. Implement exponential backoff.

## E503 Service Unavailable

The detection service is temporarily down for maintenance. Check the status
page. This resolves automatically; no action needed on your end.

## E507 Storage Quota Exceeded

Your detection history has filled your storage allocation. Upgrade your tier
or export and delete old detections to free space.
