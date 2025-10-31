# Running GitHub Pages Locally (Mac)


```bash
brew install ruby
echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
cd docs
gem install bundler
bundle install
bundle exec jekyll serve
```
