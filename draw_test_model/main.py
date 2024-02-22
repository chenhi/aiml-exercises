from app import app


###### GOOGLE CLOUD STUFF #####

# Set environment variables
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


###############################