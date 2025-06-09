# PROMPT-ENGINEERING- 1.	Comprehensive Report on the Fundamentals of Generative AI and Large Language Models (LLMs)
## Topic 1: Overview of Large Language Models (LLMs)
### Aim:
To provide a foundational understanding of LLMs, including their structure, function, and practical applications.

### Procedure:
1.Define what Large Language Models (LLMs) are and explain their role in natural language understanding and generation.

2.Describe the underlying neural network structure of LLMs, focusing on the transformer model.

3.Explain how LLMs generate human-like language from text prompts, using examples such as chatbots and text generation tools.

4.Provide examples of popular LLMs like GPT and BERT, highlighting their impact on natural language processing tasks.

5.Discuss the concepts of pre-training and fine-tuning, and how they improve the performance of LLMs on specific tasks.

6.Summary of benefits and challenges

### What are large language models (LLMs)? 

A large language model is a type of artificial intelligence algorithm that uses deep learning techniques and massively large data sets to understand, summarize, generate and predict new content. The term  generative AI also is closely connected with LLMs, which are, in fact, a type of generative AI that has been specifically architected to help generate text-based content.
Over millennia, humans developed spoken languages to communicate. Language is at the core of all forms of human and technological communications; it provides the words, semantics and grammar needed to convey ideas and concepts. In the AI world, a  language model serves a similar purpose, providing a basis to communicate and generate new concepts.The first AI language models trace their roots to the earliest days of AI. The Eliza language model debuted in 1966 at MIT and is one of the earliest examples of an AI language model. All language models are first trained on a set of data, then make use of various techniques to infer relationships before ultimately generating new content based on the trained data. Language models are commonly used in natural language processing applications where a user inputs a query in natural language to generate a result.An LLM is the evolution of the language model concept in AI that dramatically expands the data used for training and inference. In turn, it provides a massive increase in the capabilities of the AI model. While there isn't a universally accepted figure for how large the data set for training needs to be, an LLM typically has at least one billion or more parameters. Parameters are a  machine learning term for the variables present in the model on which it was trained that can be used to infer new content.

![image](https://github.com/user-attachments/assets/ebc6ef36-a0dd-46b5-81b2-e37bd528043d)

### Examples of LLMs
Here is a list of the top 10 LLMs on the market, listed in alphabetical order based on internet research:

1.Bidirectional Encoder Representations from Transformers, commonly referred to as Bert.

2.Claude.

3.Cohere.

4.Enhanced Representation through Knowledge Integration, or Ernie.

5.Falcon 40B.

6.Galactica.

7.Generative Pre-trained Transformer 3, commonly known as GPT-3.

8.GPT-3.5.

9.GPT-4.

10.Language Model for Dialogue Applications, or Lamda.

### How do large language models work?

LLMs take a complex approach that involves multiple components.
At the foundational layer, an LLM needs to be trained on a large volume -- sometimes referred to as a corpus -- of data that is typically petabytes in size. The training can take multiple steps, usually starting with an unsupervised learning approach. In that approach, the model is trained on unstructured data and unlabeled data. The benefit of training on unlabeled data is that there is often vastly more data available. At this stage, the model begins to derive relationships between different words and concepts.The next step for some LLMs is training and fine-tuning with a form of self-supervised learning. Here, some data labeling has occurred, assisting the model to more accurately identify different concepts.Next, the LLM undertakes deep learning as it goes through the transformer neural network process. The transformer model architecture enables the LLM to understand and recognize the relationships and connections between words and concepts using a self-attention mechanism. That mechanism is able to assign a score, commonly referred to as a weight, to a given item -- called a token -- in order to determine the relationship.Once an LLM has been trained, a base exists on which the AI can be used for practical purposes. By querying the LLM with a prompt, the AI model inference can generate a response, which could be an answer to a question, newly generated text, summarized text or a sentiment analysis report.

![image](https://github.com/user-attachments/assets/fa9f2d28-9252-4ed3-9d9f-5ef8bc88dcd3)

## What are large language models used for?

LLMs have become increasingly popular because they have broad applicability for a range of NLP tasks, including the following:
Text generation. The ability to generate text on any topic that the LLM has been trained on is a primary use case.
Translation. For LLMs trained on multiple languages, the ability to translate from one language to another is a common feature.
Content summary. Summarizing blocks or multiple pages of text is a useful function of LLMs.
Rewriting content. Rewriting a section of text is another capability.
Classification and categorization. An LLM is able to classify and categorize content.
Sentiment analysis. Most LLMs can be used for sentiment analysis to help users to better understand the intent of a piece of content or a particular response.
Conversational AI and chatbots. LLMs can enable a conversation with a user in a way that is typically more natural than older generations of AI technologies.
Among the most common uses for conversational AI is through a chatbot, which can exist in any number of different forms where a user interacts in a query-and-response model. The most widely used LLM-based AI chatbot is ChatGPT, which is developed by OpenAI. ChatGPT currently is based on the GPT-3.5 model, although paying subscribers can use the newer GPT-4 LLM.

### What are the advantages of large language models?

There are numerous advantages that LLMs provide to organizations and users:
Extensibility and adaptability. LLMs can serve as a foundation for customized use cases. Additional training on top of an LLM can create a finely tuned model for an organization's specific needs.
Flexibility. One LLM can be used for many different tasks and deployments across organizations, users and applications.
Performance. Modern LLMs are typically high-performing, with the ability to generate rapid, low-latency responses.
Accuracy. As the number of parameters and the volume of trained data grow in an LLM, the transformer model is able to deliver increasing levels of accuracy.
Ease of training. Many LLMs are trained on unlabeled data, which helps to accelerate the training process.
Efficiency. LLMs can save employees time by automating routine tasks.

### Why are LLMs becoming important to businesses?

As AI continues to grow, its place in the business setting becomes increasingly dominant. This is shown through the use of LLMs as well as machine learning tools. In the process of composing and applying machine learning models, research advises that simplicity and consistency should be among the main goals. Identifying the issues that must be solved is also essential, as is comprehending historical data and ensuring accuracy.
The benefits associated with machine learning are often grouped into four categories: efficiency, effectiveness, experience and business evolution. As these continue to emerge, businesses invest in this technology.

### What are the challenges and limitations of large language models?

While there are many advantages to using LLMs, there are also several challenges and limitations:
Development costs. To run, LLMs generally require large quantities of expensive graphics processing unit hardware and massive data sets.
Operational costs. After the training and development period, the cost of operating an LLM for the host organization can be very high.
Bias. A risk with any AI trained on unlabeled data is bias, as it's not always clear that known bias has been removed.
Ethical concerns. LLMs can have issues around data privacy and create harmful content.
Explainability. The ability to explain how an LLM was able to generate a specific result is not easy or obvious for users.
Hallucination. AI hallucination occurs when an LLM provides an inaccurate response that is not based on trained data.
Complexity. With billions of parameters, modern LLMs are exceptionally complicated technologies that can be particularly complex to troubleshoot.
Glitch tokens. Maliciously designed prompts that cause an LLM to malfunction, known as glitch tokens, are part of an emerging trend since 2022.
Security risks. LLMs can be used to improve phishing attacks on employees.

## What are the different types of large language models?
There is an evolving set of terms to describe the different types of large language models. Among the common types are the following:
Zero-shot model. This is a large, generalized model trained on a generic corpus of data that is able to give a fairly accurate result for general use cases, without the need for additional training. GPT-3 is often considered a zero-shot model.
Fine-tuned or domain-specific models. Additional training on top of a zero-shot model such as GPT-3 can lead to a fine-tuned, domain-specific model. One example is OpenAI Codex, a domain-specific LLM for programming based on GPT-3.
Language representation model. One example of a language representation model is Google's Bert, which makes use of deep learning and transformers well suited for NLP.
Multimodal model. Originally LLMs were specifically tuned just for text, but with the multimodal approach it is possible to handle both text and images. GPT-4 is an example of this type of model.

### The future of large language models

![image](https://github.com/user-attachments/assets/9789cda4-2f4e-40d4-9d7a-f769fe8b62ec)

The future of LLMs is still being written by the humans who are developing the technology, though there could be a future in which the LLMs write themselves, too. The next generation of LLMs will not likely be artificial general intelligence or sentient in any sense of the word, but they will continuously improve and get "smarter."LLMs will also continue to expand in terms of the business applications they can handle. Their ability to translate content across different contexts will grow further, likely making them more usable by business users with different levels of technical expertise.LLMs will continue to be trained on ever larger sets of data, and that data will increasingly be better filtered for accuracy and potential bias, partly through the addition of fact-checking capabilities. It's also likely that LLMs of the future will do a better job than the current generation when it comes to providing attribution and better explanations for how a given result was generated.Enabling more accurate information through domain-specific LLMs developed for individual industries or functions is another possible direction for the future of large language models. Expanded use of techniques such as reinforcement learning from human feedback, which OpenAI uses to train ChatGPT, could help improve the accuracy of LLMs too. There's also a class of LLMs based on the concept known as retrieval-augmented generation -- including Google's Realm, which is short for Retrieval-Augmented Language Model -- that will enable training and inference on a very specific corpus of data, much like how a user today can specifically search content on a single site.
There's also ongoing work to optimize the overall size and training time required for LLMs, including development of Meta's Llama model. Llama 2, which was released in July 2023, has less than half the parameters than GPT-3 has and a fraction of the number GPT-4 contains, though its backers claim it can be more accurate.On the other hand, the use of large language models could drive new instances of shadow IT in organizations. CIOs will need to implement usage guardrails and provide training to avoid data privacy problems and other issues. LLMs could also create new cybersecurity challenges by enabling attackers to write more persuasive and realistic phishing emails or other malicious communications.Nonetheless, the future of LLMs will likely remain bright as the technology continues to evolve in ways that help improve human productivity.

### Overview of large language models:

![image](https://github.com/user-attachments/assets/b8728a24-5434-4338-96e7-70f991da9c2f)

### Topic 2: Introduction to Generative AI
### Aim:
To introduce the concept of Generative AI, explain how it works, and discuss its applications and challenges.

### Procedure:
1.Define Generative AI and outline its key characteristics.
2.Illustrate the process by which Generative AI creates new data (e.g., text, images, or music).
3.Identify real-world applications of Generative AI in fields like healthcare, entertainment, and content creation.
4.Discuss the advantages and challenges of Generative AI, focusing on creative automation, efficiency, and ethical concerns.
5.Summary of benefits and challenges

### What is Generative AI?
Generative AI enables users to quickly generate new content based on a variety of inputs. Inputs and outputs to these models can include text, images, sounds, animation, 3D models, or other types of data.

### How Does Generative AI Work?
Generative AI models use neural networks to identify the patterns and structures within existing data to generate new and original content.
One of the breakthroughs with generative AI models is the ability to leverage different learning approaches, including unsupervised or semi-supervised learning for training. This has given organizations the ability to more easily and quickly leverage a large amount of unlabeled data to create foundation models. As the name suggests, foundation models can be used as a base for AI systems that can perform multiple tasks. 
Examples of foundation models include GPT-3 and Stable Diffusion, which allow users to leverage the power of language. For example, popular applications like ChatGPT, which draws from GPT-3, allow users to generate an essay based on a short text request. On the other hand, Stable Diffusion allows users to generate photorealistic images given a text input.

### How to Evaluate Generative AI Models?
1.Quality: Especially for applications that interact directly with users, having high-quality generation outputs is key. For example, in speech generation, poor speech quality is difficult to understand. Similarly, in image generation, the desired outputs should be visually indistinguishable from natural images.
2.Diversity: A good generative model captures the minority modes in its data distribution without sacrificing generation quality. This helps reduce undesired biases in the learned models.
3.Speed: Many interactive applications require fast generation, such as real-time image editing to allow use in content creation workflows.

![image](https://github.com/user-attachments/assets/e4d73ba5-a502-4de6-8499-1ddc855a28ea)

### How to Develop Generative AI Models?
There are multiple types of generative models, and combining the positive attributes of each results in the ability to create even more powerful models.
Below is a breakdown:
Diffusion models: Also known as denoising diffusion probabilistic models (DDPMs), diffusion models are generative models that determine vectors in latent space through a two-step process during training. The two steps are forward diffusion and reverse diffusion. The forward diffusion process slowly adds random noise to training data, while the reverse process reverses the noise to reconstruct the data samples. Novel data can be generated by running the reverse denoising process starting from entirely random noise.

![image](https://github.com/user-attachments/assets/a447cf47-5a92-40b4-b1ec-3393450292ac)

A diffusion model can take longer to train than a variational autoencoder (VAE) model, but thanks to this two-step process, hundreds, if not an infinite amount, of layers can be trained, which means that diffusion models generally offer the highest-quality output when building generative AI models.
Additionally, diffusion models are also categorized as foundation models, because they are large-scale, offer high-quality outputs, are flexible, and are considered best for generalized use cases. However, because of the reverse sampling process, running foundation models is a slow, lengthy process.
Variational autoencoders (VAEs): VAEs consist of two neural networks typically referred to as the encoder and decoder.
When given an input, an encoder converts it into a smaller, more dense representation of the data. This compressed representation preserves the information that’s needed for a decoder to reconstruct the original input data, while discarding any irrelevant information. The encoder and decoder work together to learn an efficient and simple latent data representation. This allows the user to easily sample new latent representations that can be mapped through the decoder to generate novel data.
While VAEs can generate outputs such as images faster, the images generated by them are not as detailed as those of diffusion models.
Generative adversarial networks (GANs): Discovered in 2014, GANs were considered to be the most commonly used methodology of the three before the recent success of diffusion models. GANs pit two neural networks against each other: a generator that generates new examples and a discriminator that learns to distinguish the generated content as either real (from the domain) or fake (generated).
The two models are trained together and get smarter as the generator produces better content and the discriminator gets better at spotting the generated content. This procedure repeats, pushing both to continually improve after every iteration until the generated content is indistinguishable from the existing content.
While GANs can provide high-quality samples and generate outputs quickly, the sample diversity is weak, therefore making GANs better suited for domain-specific data generation.
Another factor in the development of generative models is the architecture underneath. One of the most popular is the transformer network. It is important to understand how it works in the context of generative AI.
Transformer networks: Similar to recurrent neural networks, transformers are designed to process sequential input data non-sequentially.
Two mechanisms make transformers particularly adept for text-based generative AI applications: self-attention and positional encodings. Both of these technologies help represent time and allow for the algorithm to focus on how words relate to each other over long distances

![image](https://github.com/user-attachments/assets/cf6510b7-a109-4157-ba06-3375bcca9b10)

A self-attention layer assigns a weight to each part of an input. The weight signifies the importance of that input in context to the rest of the input. Positional encoding is a representation of the order in which input words occur.
A transformer is made up of multiple transformer blocks, also known as layers. For example, a transformer has self-attention layers, feed-forward layers, and normalization layers, all working together to decipher and predict streams of tokenized data, which could include text, protein sequences, or even patches of images.

### What are the Applications of Generative AI?
Generative AI is a powerful tool for streamlining the workflow of creatives, engineers, researchers, scientists, and more. The use cases and possibilities span all industries and individuals.Generative AI models can take inputs such as text, image, audio, video, and code and generate new content into any of the modalities mentioned. For example, it can turn text inputs into an image, turn an image into a song, or turn video into text.

![image](https://github.com/user-attachments/assets/49bba533-4073-4ca5-9698-0ccb66fbbf92)

### Here are the most popular generative AI applications:
Language: Text is at the root of many generative AI models and is considered to be the most advanced domain. One of the most popular examples of language-based generative models are called large language models (LLMs). Large language models are being leveraged for a wide variety of tasks, including essay generation, code development, translation, and even understanding genetic sequences.

Audio: Music, audio, and speech are also emerging fields within generative AI. Examples include models being able to develop songs and snippets of audio clips with text inputs, recognize objects in videos and create accompanying noises for different video footage, and even create custom music.

Visual: One of the most popular applications of generative AI is within the realm of images. This encompasses the creation of 3D images, avatars, videos, graphs, and other illustrations. There’s flexibility in generating images with different aesthetic styles, as well as techniques for editing and modifying generated visuals. Generative AI models can create graphs that show new chemical compounds and molecules that aid in drug discovery, create realistic images for virtual or augmented reality, produce 3D models for video games, design logos, enhance or edit existing images, and more.

Synthetic data: Synthetic data is extremely useful to train AI models when data doesn’t exist, is restricted, or is simply unable to address corner cases with the highest accuracy. The development of synthetic data through generative models is perhaps one of the most impactful solutions for overcoming the data challenges of many enterprises. It spans all modalities and use cases and is possible through a process called label efficient learning. Generative AI models can reduce labeling costs by either automatically producing additional augmented training data or by learning an internal representation of the data that facilitates training AI models with less labeled data.

The impact of generative models is wide-reaching, and its applications are only growing. Listed are just a few examples of how generative AI is helping to advance and transform the fields of transportation, natural sciences, and entertainment.

In the automotive industry, generative AI is expected to help create 3D worlds and models for simulations and car development. Synthetic data is also being used to train autonomous vehicles. Being able to road test the abilities of an autonomous vehicle in a realistic 3D world improves safety, efficiency, and flexibility while decreasing risk and overhead.

The field of natural sciences greatly benefits from generative AI. In the healthcare industry, generative models can aid in medical research by developing new protein sequences to aid in drug discovery. 

Practitioners can also benefit from the automation of tasks such as scribing, medical coding, medical imaging, and genomic analysis. Meanwhile, in the weather industry, generative models can be used to create simulations of the planet and help with accurate weather forecasting and natural disaster prediction. These applications can help to create safer environments for the general population and allow scientists to predict and better prepare for natural disasters.

All aspects of the entertainment industry, from video games to film, animation, world building, and virtual reality, are able to leverage generative AI models to help streamline their content creation process. Creators are using generative models as a tool to help supplement their creativity and work.

### What are the Challenges of Generative AI?
As an evolving space, generative models are still considered to be in their early stages, giving them space for growth in the following areas.

1.Scale of compute infrastructure: Generative AI models can boast billions of parameters and require fast and efficient data pipelines to train. Significant capital investment, technical expertise, and large-scale compute infrastructure are necessary to maintain and develop generative models. For example, diffusion models could require millions or billions of images to train. Moreover, to train such large datasets, massive compute power is needed, and AI practitioners must be able to procure and leverage hundreds of GPUs to train their models.

2.Sampling speed: Due to the scale of generative models, there may be latency present in the time it takes to generate an instance. Particularly for interactive use cases such as chatbots, AI voice assistants, or customer service applications, conversations must happen immediately and accurately. As diffusion models become increasingly popular due to the high-quality samples that they can create, their slow sampling speeds have become increasingly apparent.

3.Lack of high-quality data: Oftentimes, generative AI models are used to produce synthetic data for different use cases. However, while troves of data are being generated globally every day, not all data can be used to train AI models. Generative models require high-quality, unbiased data to operate. Moreover, some domains don’t have enough data to train a model. As an example, few 3D assets exist and they’re expensive to develop. Such areas will require significant resources to evolve and mature.

4.Data licenses: Further compounding the issue of a lack of high-quality data, many organizations struggle to get a commercial license to use existing datasets or to build bespoke datasets to train generative models. This is an extremely important process and key to avoiding intellectual property infringement issues.

Many companies such as NVIDIA, Cohere, and Microsoft have a goal to support the continued growth and development of generative AI models with services and tools to help solve these issues. These products and platforms abstract away the complexities of setting up the models and running them at scale.

### What are the Benefits of Generative AI?
Generative AI is important for a number of reasons. Some of the key benefits of generative AI include:

1.Generative AI algorithms can be used to create new, original content, such as images, videos, and text, that’s indistinguishable from content created by humans. This can be useful for applications such as entertainment, advertising, and creative arts.

2.Generative AI algorithms can be used to improve the efficiency and accuracy of existing AI systems, such as natural language processing and computer vision. For example, generative AI algorithms can be used to create synthetic data that can be used to train and evaluate other AI algorithms.

3.Generative AI algorithms can be used to explore and analyze complex data in new ways, allowing businesses and researchers to uncover hidden patterns and trends that may not be apparent from the raw data alone.

4.Generative AI algorithms can help automate and accelerate a variety of tasks and processes, saving time and resources for businesses and organizations. Generative AI offers numerous advantages across various industries, enhancing productivity, creativity, and efficiency. One of its key benefits is its ability to automate content creation, generating high-quality text, images, music, and even videos. 

5.This makes it invaluable in industries like marketing, journalism, and entertainment, where rapid content production is essential. Additionally, it significantly improves productivity by assisting with tasks such as summarization, translation, and coding, reducing human effort and allowing professionals to focus on higher-level decision-making. Another major advantage is personalization, as AI-powered systems can tailor responses and recommendations based on user preferences, enhancing customer experiences in areas like e-commerce and customer support. 

6.Generative AI also fosters creativity by assisting artists, writers, and designers in brainstorming and producing novel ideas. Moreover, it helps businesses reduce costs by automating repetitive tasks, improving operational efficiency, and enabling scalability through AI-driven solutions like virtual assistants and chatbots.

7.Privacy and security risks are another major issue, as AI models can inadvertently expose sensitive data, and technologies like deepfakes can be used for misinformation or fraud. Moreover, the increasing reliance on AI may contribute to job displacement in certain industries, as automation replaces roles traditionally performed by humans.

8.Legal and regulatory frameworks for AI-generated content remain unclear, leading to potential intellectual property disputes and ethical dilemmas. Additionally, generative AI lacks true common sense and reasoning, meaning it can struggle with complex decision-making and logical analysis. While it is a powerful tool, ensuring responsible and ethical AI development is crucial to mitigating these risks and maximizing its benefits.

Overall, generative AI has the potential to significantly impact a wide range of industries and applications and is an important area of AI research and development. 
