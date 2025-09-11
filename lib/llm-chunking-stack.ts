import * as cdk from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as sqs from "aws-cdk-lib/aws-sqs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambdaEventSources from "aws-cdk-lib/aws-lambda-event-sources";
import { Construct } from "constructs";

export class LlmChunkingStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // DynamoDB table for job tracking
    const jobTable = new dynamodb.Table(this, "JobTable", {
      partitionKey: { name: "job_id", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // S3 buckets
    const rejectedChunksBucket = new s3.Bucket(this, "RejectedChunksBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const ingestedChunksBucket = new s3.Bucket(this, "IngestedChunksBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const textractOutputBucket = new s3.Bucket(this, "TextractOutputBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // SQS queues
    const dlq = new sqs.Queue(this, "ChunkDLQ", {
      retentionPeriod: cdk.Duration.days(14),
    });

    const chunkQueue = new sqs.Queue(this, "ChunkQueue", {
      deadLetterQueue: {
        queue: dlq,
        maxReceiveCount: 3,
      },
      visibilityTimeout: cdk.Duration.minutes(15),
    });

    // Lambda layer for shared Pydantic models
    const sharedLayer = new lambda.LayerVersion(this, "SharedLayer", {
      code: lambda.Code.fromAsset("lambda/layers/shared", {
        bundling: {
          image: lambda.Runtime.PYTHON_3_13.bundlingImage,
          command: [
            "bash",
            "-c",
            "pip install --platform manylinux2014_x86_64 --only-binary=:all: --prefer-binary -r requirements.txt -t /asset-output/python && cp -r . /asset-output/python",
          ],
        },
      }),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_13],
    });

    // Lambda layer for Amazon Textract
    const textractLayer = new lambda.LayerVersion(this, "TextractLayer", {
      code: lambda.Code.fromAsset("lambda/layers/textract", {
        bundling: {
          image: lambda.Runtime.PYTHON_3_13.bundlingImage,
          command: [
            "bash",
            "-c",
            "pip install --platform manylinux2014_x86_64 --only-binary=:all: --prefer-binary -r requirements.txt -t /asset-output/python && cp -r . /asset-output/python",
          ],
        },
      }),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_13],
    });

    // LLM inference lambda
    const llmLambda = new lambda.Function(this, "LlmInferenceLambda", {
      runtime: lambda.Runtime.PYTHON_3_13,
      handler: "handler.handler",
      code: lambda.Code.fromAsset("lambda/llm-inference", {
        bundling: {
          image: lambda.Runtime.PYTHON_3_13.bundlingImage,
          command: [
            "bash",
            "-c",
            "pip install --platform manylinux2014_x86_64 --only-binary=:all: --prefer-binary -r requirements.txt -t /asset-output && cp -au . /asset-output",
          ],
        },
      }),
      timeout: cdk.Duration.minutes(15),
      memorySize: 1024,
      layers: [sharedLayer],
      environment: {
        JOB_TABLE_NAME: jobTable.tableName,
        REJECTED_BUCKET_NAME: rejectedChunksBucket.bucketName,
        INGESTED_BUCKET_NAME: ingestedChunksBucket.bucketName,
        CHUNK_QUEUE_URL: chunkQueue.queueUrl,
        DLQ_URL: dlq.queueUrl,
      },
    });

    // Ingestion script lambda
    const ingestionLambda = new lambda.Function(this, "IngestionLambda", {
      runtime: lambda.Runtime.PYTHON_3_13,
      handler: "handler.handler",
      code: lambda.Code.fromAsset("lambda/ingestion", {
        bundling: {
          image: lambda.Runtime.PYTHON_3_13.bundlingImage,
          command: [
            "bash",
            "-c",
            "pip install --platform manylinux2014_x86_64 --only-binary=:all: --prefer-binary -r requirements.txt -t /asset-output && cp -au . /asset-output",
          ],
        },
      }),
      timeout: cdk.Duration.minutes(15),
      memorySize: 512,
      layers: [sharedLayer, textractLayer],
      environment: {
        JOB_TABLE_NAME: jobTable.tableName,
        CHUNK_QUEUE_URL: chunkQueue.queueUrl,
        TEXTRACT_OUTPUT_BUCKET: textractOutputBucket.bucketName,
      },
    });
    // Event source mapping with batch size 1
    llmLambda.addEventSource(
      new lambdaEventSources.SqsEventSource(chunkQueue, {
        batchSize: 1,
      }),
    );

    // Permissions
    jobTable.grantReadWriteData(llmLambda);
    jobTable.grantReadWriteData(ingestionLambda);
    rejectedChunksBucket.grantWrite(llmLambda);
    ingestedChunksBucket.grantWrite(llmLambda);
    textractOutputBucket.grantReadWrite(ingestionLambda);
    chunkQueue.grantConsumeMessages(llmLambda);
    chunkQueue.grantSendMessages(ingestionLambda);
    dlq.grantSendMessages(llmLambda);

    // Grant ingestion lambda comprehensive S3 permissions
    ingestionLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:ListObjectsV2",
          "s3:DeleteObject",
        ],
        resources: [
          "arn:aws:s3:::wisconsin-chatbot-sources-west-2",
          "arn:aws:s3:::wisconsin-chatbot-sources-west-2/*",
          "arn:aws:s3:::chunks-output-bucket/*",
          "arn:aws:s3:::logs-output-bucket/*",
          textractOutputBucket.bucketArn,
          `${textractOutputBucket.bucketArn}/*`,
        ],
      }),
    );

    // Bedrock permissions for LLM lambda
    llmLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["bedrock:InvokeModel"],
        resources: ["*"],
      }),
    );
  }
}
