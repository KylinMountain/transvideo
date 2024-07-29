import logging
import os
import sys

import openai
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

client = openai.OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                       base_url="https://api.groq.com/openai/v1")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def get_completion(user_prompt, system_message=""):
    logging.info(f"request: {system_message}, user_prompt: {user_prompt}")
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        temperature=0.1,
        max_tokens=4000,
        messages=[
            ChatCompletionSystemMessageParam(
                content=system_message,
                role="system"
            ),
            ChatCompletionUserMessageParam(
                content=user_prompt,
                role="user"
            )
        ]
    )
    logging.info(f"response: {response.choices[0].message.content}")

    return response.choices[0].message.content


def initial_translation(source_text):
    system_message = f"你是一个专业的翻译官，尤其擅长科技类英文到中文翻译。"
    translation_prompt = f"""你的任务是将英文字幕翻译为中文字幕, 必须保持原本的字幕格式，要求保持精简，翻译字幕和要与原文基本一致，不要肆意添加任何解释或者其他文字，不要漏翻。
字幕中一句话通常分布在几个字幕块中，翻译时候，务必整句翻译，并将其按照原格式分布在多个字幕块，保持字符数量和字幕时间段一致。
举个例子：
英文字幕：
8
00:00:25,514 --> 00:00:29,257
She's also
contributed to the OpenAI Cookbook that

9
00:00:29,257 --> 00:00:30,958
teaches people prompting.
So thrilled

10
00:00:30,958 --> 00:00:32,660
to have you with me.
中文字幕：
8
00:00:25,514 --> 00:00:29,257
她还为OpenAI Cookbook做出了贡献，

9
00:00:29,257 --> 00:00:30,958
专门教人们如何进行prompting。很高兴

10
00:00:30,958 --> 00:00:32,660
你能与我一起。

开始你的翻译：
英文字幕: {source_text}

中文字幕:"""

    translation = get_completion(
        translation_prompt, system_message
    )
    return translation


def reflect_on_translation(source_text, translation):
    system_message = f"你是一名翻译专家，专门从事从英文字幕到中文字幕的翻译工作。你将得到一段源文本及其翻译，你的任务是改进这段翻译。"

    reflection_prompt = f"""你的任务是仔细阅读从英文字幕翻译为中文字幕，然后提出建设性的批评和有帮助的建议，以改进翻译。

源文本和初始翻译如下，以XML标签<SOURCE_TEXT></SOURCE_TEXT>和分隔：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation}
</TRANSLATION>

在写建议时，请注意是否有以下改进翻译的方法：
(i) 准确性（通过纠正添加错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用中文的语法、拼写和标点规则，确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格，并考虑任何文化背景），
(iv) 术语（确保术语使用一致，并反映源文本的领域；并确保仅使用等效的中文术语）。
(v) 针对字幕本身具有时效性，切勿增长单个字幕快的内容长度，保持前后字幕连贯。
写出一份具体、有帮助和建设性的建议清单，以改进翻译。
每个建议应解决翻译中的一个特定部分。
只输出建议，不要包含其他内容。
"""
    prompt = reflection_prompt.format(
        source_text=source_text,
        translation=translation,
    )
    reflection = get_completion(prompt, system_message=system_message)
    return reflection


def improve_translation(source_text, translation, reflection):
    system_message = f"你是一个专业的翻译官，尤其擅长科技类英文到中文翻译。"

    prompt = f"""你的任务是仔细阅读并编辑从英文字幕到中文字幕的翻译，考虑专家的建议和建设性的批评。
针对字幕本身具有时效性，必须考虑原始英文字幕和中文字幕块的一致性，不要导致字幕本身的时效性变化，不要增加单个字幕快的内容长度，务必保持前后字幕连贯，前后字幕不要重复翻译！！！干得好，奖励5000美元。

源英文字幕、初始翻译和专家语言学家的建议如下，以XML标签<SOURCE_TEXT></SOURCE_TEXT>、和<EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS>分隔：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

请在编辑翻译时考虑专家的建议。编辑翻译时请确保：

(i) 准确性（通过纠正添加错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用中文的语法、拼写和标点规则，确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格，并考虑任何文化背景），
(iv) 术语（确保术语使用一致，并反映源文本的领域；并确保仅使用等效的中文术语）。
(v) 针对字幕本身具有时效性，切勿增长单个字幕快的内容长度，保持前后字幕连贯。

只输出新的字幕翻译，不要包含其他内容，如果无需修改，返回原始翻译，如果移除内容，就展示为空，不要有任何提示内容。

"""

    translation_2 = get_completion(prompt, system_message)

    return translation_2


def translate_agent(source_text):
    translation_1 = initial_translation(
        source_text
    )

    reflection = reflect_on_translation(
        source_text, translation_1
    )
    translation_2 = improve_translation(
        source_text, translation_1, reflection
    )

    return translation_2
