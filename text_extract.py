# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request

app = Flask(__name__)
"""SparkOcr.ipynb
Spark OCR

## Spark OCR transformers and Spark NLP annotators
"""

secret = "xxxx"
license = "xxx"
version = secret.split("-")[0]
spark_ocr_jar_path = "../../target/scala-2.11"


import os
import sys


"""## Initialization of spark session
Need specify path to `spark-ocr-assembly.jar` or `secret`
"""
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
from sparkocr import start

if license:
    os.environ['JSL_OCR_LICENSE'] = license

spark = start(secret=secret, jar_path=spark_ocr_jar_path, nlp_version="2.4.5")
spark

from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from sparkocr.transformers import *
from sparknlp.annotator import *
from sparknlp.base import *
from sparkocr.enums import PageSegmentationMode

"""## Define OCR transformers and pipeline"""

def update_text_pipeline():

    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    tokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("tokens")

    spell = NorvigSweetingModel().pretrained("spellcheck_norvig", "en") \
          .setInputCols("tokens") \
          .setOutputCol("spell")
    
    tokenAssem = TokenAssembler() \
          .setInputCols("spell") \
          .setOutputCol("newDocs")

    updatedText = UpdateTextPosition() \
          .setInputCol("positions") \
          .setOutputCol("output_positions") \
          .setInputText("newDocs.result")

    pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        spell,
        tokenAssem,
        updatedText
    ])
    
    return pipeline


def ocr_pipeline():
    # Transforrm PDF document to images per page
        pdf_to_image = PdfToImage() \
            .setInputCol("content") \
            .setOutputCol("image_raw") \
            .setKeepInput(True)

        # adaptive_thresholding = ImageAdaptiveThresholding() \
        #     .setInputCol("image_raw") \
        #     .setOutputCol("corrected_image") \
        #     .setBlockSize(35) \
        #     .setOffset(80)

        binarizer = ImageBinarizer() \
            .setInputCol("image_raw") \
            .setOutputCol("image") \
            .setThreshold(130)

        ocr = ImageToText() \
            .setInputCol("image") \
            .setOutputCol("text") \
            .setIgnoreResolution(False) \
            .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
            .setConfidenceThreshold(60)

        pipeline = Pipeline(stages=[
            pdf_to_image,
            binarizer,
            ocr
        ])
        return pipeline

"""## Read PDF document as binary file"""
@app.route("/file", methods=['GET','POST'])
def extract_pdf():
    file = str(request.files['document'].read(), 'utf-8')
#    imagePath = file
    df = spark.read \
        .format("binaryFile") \
        .load(file).cache()



    """## Run OCR pipelines"""

    ocr_result = ocr_pipeline().fit(df).transform(df)
    result= update_text_pipeline().fit(ocr_result).transform(ocr_result)
    res = result.toPandas()

    results = res[["path", "pagenum", "confidence", "text", "document", "newDocs", "spell"]]
    return jsonify(results)



if __name__ == "__main__":
    app.run(debug=True)
