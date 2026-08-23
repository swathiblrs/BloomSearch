# User-owned Google Cloud Run deployment

This project does not create or operate a shared BloomSearch service. Each user deploys it into their own Google Cloud project and is responsible for Google Cloud and hosted-model API charges.

## Before deploying

1. Create or select your own Google Cloud project.
2. Enable billing in that project if Google requires it.
3. Install and authenticate the `gcloud` CLI.
4. Review current Cloud Run and Cloud Build pricing.
5. Create a budget and billing alerts in Google Cloud Billing.

## Deploy from source

Run from the repository root, replacing the project ID:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud run deploy bloomsearch \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1 \
  --cpu 1 \
  --memory 512Mi \
  --timeout 120
```

The command displays the service URL when deployment finishes. The user enters their own compatible API base URL, API key, and hosted model ID in the web page.

## Cost ownership

- The Google Cloud project owner pays any Cloud Run, Cloud Build, Artifact Registry, logging, networking, or related charges.
- The API-key owner pays any hosted-model charges.
- The original repository owner does not receive, store, proxy, or pay for those credentials or services.
- `--min-instances 0` allows scaling to zero, but it does not guarantee a zero bill.
- `--max-instances 1` limits scaling; it is not a spending guarantee.

## Delete the deployment

Deleting unused resources prevents continuing charges:

```bash
gcloud run services delete bloomsearch --region us-central1
```

Users should also inspect and delete unneeded Cloud Build artifacts and Artifact Registry images in their own project.

